import numpy as np
import time as time_module
from collections import defaultdict

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: 150s total, use 140s for dispatch
    deadline = time_module.time() + 140.0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Precompute features
    features = []
    for op in ready_operations:
        durations = list(op["machine_options"].values())
        min_dur = min(durations) if durations else 0
        max_dur = max(durations) if durations else 0
        avg_dur = sum(durations) / len(durations) if durations else 0
        n_machines = len(durations) if durations else 1
        rem_ops = op.get("remaining_ops", [])
        n_rem = len(rem_ops) if isinstance(rem_ops, list) else 0
        rw = op["remaining_work"]

        features.append({
            "remaining_work": rw,
            "min_dur": min_dur,
            "max_dur": max_dur,
            "avg_dur": avg_dur,
            "n_machines": n_machines,
            "n_rem_ops": n_rem,
        })

    # Per-job remaining work
    job_rw = defaultdict(float)
    for i, op in enumerate(ready_operations):
        jid = op["job"]
        job_rw[jid] = max(job_rw[jid], op["remaining_work"])

    # Check if simulation is possible
    can_simulate = False
    sample_op = ready_operations[0]
    rem_ops = sample_op.get("remaining_ops", None)
    if isinstance(rem_ops, list) and len(rem_ops) > 0:
        if isinstance(rem_ops[0], dict) and "machine_options" in rem_ops[0]:
            can_simulate = True

    # Smooth max via log-sum-exp
    def soft_max(values, beta=15.0):
        if not values:
            return 0.0
        arr = np.array(values, dtype=np.float64)
        m = np.max(arr)
        if m == 0:
            return 0.0
        return float(m + np.log(np.sum(np.exp(beta * (arr - m)))) / beta)

    def estimate_makespan(ready_ops, current_time):
        """Estimate makespan using both job-level and machine-level load balance."""
        if not ready_ops:
            return current_time
        # Job-level remaining work
        jr = defaultdict(float)
        # Machine-level estimated work (sum of min durations per machine)
        mr = defaultdict(float)
        for op in ready_ops:
            jid = op["job"]
            jr[jid] = max(jr[jid], op["remaining_work"])
            # For machine load, use the minimum duration (best-case assignment)
            durations = list(op["machine_options"].values())
            min_dur = min(durations) if durations else 0
            # Distribute to the machine with fewest options (most constrained)
            # Use min duration as proxy for machine load contribution
            if durations:
                # Assign to the machine with smallest duration (most likely choice)
                min_m = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
                mr[min_m] += min_dur
        # Combine: makespan >= max(job_tail, machine_load)
        job_load = soft_max(list(jr.values()), beta=15.0) if jr else 0.0
        mach_load = soft_max(list(mr.values()), beta=15.0) if mr else 0.0
        # The makespan is at least the max of both, but they can overlap
        # Use a smooth max of the two as a proxy
        return current_time + soft_max([job_load, mach_load], beta=20.0)

    if can_simulate and n <= 12:
        def simulate_step(ready_ops, t, idx):
            op = ready_ops[idx]
            durations = list(op["machine_options"].values())
            min_dur = min(durations) if durations else 0
            comp_t = t + min_dur
            new_ready = []
            for j, other in enumerate(ready_ops):
                if j == idx:
                    rem = op.get("remaining_ops", [])
                    if isinstance(rem, list) and len(rem) > 0:
                        nxt = rem[0].copy()
                        nxt["remaining_work"] = max(0, op["remaining_work"] - min_dur)
                        nxt["remaining_ops"] = rem[1:] if len(rem) > 1 else []
                        new_ready.append(nxt)
                else:
                    new_ready.append(other)
            return new_ready, comp_t

        # LPT-inspired: prioritize operations with large min_duration
        # Score each candidate by a combination of LPT priority and makespan estimate
        beam_width = 5
        candidates = []
        for i in range(n):
            if time_module.time() > deadline:
                break
            new_ready, new_t = simulate_step(ready_operations, time, i)
            makespan_est = estimate_makespan(new_ready, new_t)
            # LPT priority: prefer ops with larger min_duration (longer, less flexible)
            lpt_score = features[i]["min_dur"]
            # Combined score: minimize makespan but break ties by LPT (larger min_dur first)
            # Use a weighted combination where makespan dominates but LPT breaks ties
            combined = makespan_est - 0.01 * lpt_score
            candidates.append((combined, makespan_est, i))

        if not candidates:
            return int(np.argmax([f["min_dur"] for f in features]))

        candidates.sort(key=lambda x: x[0])
        top_k = candidates[:beam_width]

        best_score = float('inf')
        best_idx = top_k[0][2]

        for l1_combined, l1_makespan, l1_idx in top_k:
            if time_module.time() > deadline:
                break
            new_ready, new_t = simulate_step(ready_operations, time, l1_idx)
            for j in range(len(new_ready)):
                if time_module.time() > deadline:
                    break
                nr2, nt2 = simulate_step(new_ready, new_t, j)
                s2 = estimate_makespan(nr2, nt2)
                # Use smooth max to combine the two-level estimates
                total = soft_max([l1_makespan, s2], beta=20.0)
                if total < best_score:
                    best_score = total
                    best_idx = l1_idx

        return best_idx

    # Fallback: LPT-inspired weighted heuristic
    # LPT prioritizes operations with large min_duration to balance machine loads early
    weight_sets = [
        # (w_min_dur, w_rw, w_rem_ops, w_flexibility)
        (1.0, 0.3, 0.2, 0.1),
        (1.0, 0.5, 0.1, 0.0),
        (0.8, 0.4, 0.3, 0.2),
        (1.0, 0.2, 0.3, 0.1),
        (0.9, 0.3, 0.2, 0.2),
        (1.0, 0.4, 0.1, 0.1),
        (1.2, 0.2, 0.2, 0.0),
        (1.0, 0.3, 0.3, 0.1),
    ]

    best_overall = float('inf')
    best_choice = 0

    for w1, w2, w3, w4 in weight_sets:
        if time_module.time() > deadline:
            break
        scores = []
        for f in features:
            # LPT: prioritize large min_duration
            lpt_term = w1 * f["min_dur"]
            # Remaining work: complete jobs with more work
            rw_term = w2 * f["remaining_work"]
            # Depth: prioritize jobs with more remaining operations
            depth_term = w3 * f["n_rem_ops"]
            # Flexibility: prefer ops with more machine options (less constrained)
            flex_term = w4 * f["n_machines"]
            s = lpt_term + rw_term + depth_term + flex_term
            scores.append(s)
        chosen = int(np.argmax(scores))
        chosen_min = features[chosen]["min_dur"]
        comp_t = time + chosen_min
        chosen_job = ready_operations[chosen]["job"]
        chosen_rw_after = max(0, features[chosen]["remaining_work"] - chosen_min)
        # Estimate makespan using both job and machine loads
        jwa = {}
        mwa = defaultdict(float)
        for i, f in enumerate(features):
            jid = ready_operations[i]["job"]
            if i == chosen:
                jwa[jid] = chosen_rw_after
            else:
                if jid in jwa:
                    jwa[jid] = max(jwa[jid], f["remaining_work"])
                else:
                    jwa[jid] = f["remaining_work"]
            # Machine load estimate
            durations = list(ready_operations[i]["machine_options"].values())
            if durations:
                min_m = min(ready_operations[i]["machine_options"].keys(), key=lambda m: ready_operations[i]["machine_options"][m])
                mwa[min_m] += f["min_dur"]
        job_load = soft_max(list(jwa.values()), beta=15.0) if jwa else 0.0
        mach_load = soft_max(list(mwa.values()), beta=15.0) if mwa else 0.0
        est = comp_t + soft_max([job_load, mach_load], beta=20.0)
        if est < best_overall:
            best_overall = est
            best_choice = chosen

    return best_choice

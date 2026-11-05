"""Job-shop dispatch rule: Critical-path urgency with machine-contention tiebreak."""
# EVOLVE-BLOCK-START
import time as _time
import math

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 140.0  # safety margin well before the 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def _soft_min(values, temperature=1.0):
    """Smooth minimum via log-sum-exp trick: -log(sum(exp(-v/T)))*T."""
    if not values:
        return 0.0
    vmin = min(values)
    if vmin == 0:
        return 0.0
    s = sum(math.exp(-(v - vmin) / temperature) for v in values)
    return vmin - temperature * math.log(s)


def _soft_max(values, temperature=1.0):
    """Smooth maximum via log-sum-exp: log(sum(exp(v/T)))*T."""
    if not values:
        return 0.0
    vmax = max(values)
    if vmax == 0:
        return 0.0
    s = sum(math.exp((v - vmax) / temperature) for v in values)
    return vmax + temperature * math.log(s)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.

    Critical-path urgency rule:
      For each ready op, estimate the critical path through its job:
        cp(i) = min_duration_current_op(i) + remaining_work(i)
      This captures both the immediate bottleneck (shortest possible duration
      of the current op) and the downstream work it gates.

      score(i) = cp(i)                        # critical-path term (dominant)
               + 1.0 / (remaining_work(i)+1)  # starvation guard
               - eta * normalized_contention  # prefer light machine slot

    Return index of the max score.
    """
    if not ready_operations:
        return 0
    if _deadline_exceeded():
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # How many ready ops target each machine (contention proxy).
    machine_counts = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo:
            machine_counts[m] = machine_counts.get(m, 0) + 1

    # Normalization range for contention (0 -> best, 1 -> worst).
    max_contention = max(machine_counts.values()) if machine_counts else 1
    denom = max(1, max_contention - 1)

    eta = 0.12  # mild contention penalty
    starve_weight = 1.0

    best_idx = 0
    best_score = float("-inf")

    for i in range(n):
        op = ready_operations[i]
        rw = float(op.get("remaining_work", 0) or 0)
        mo = op.get("machine_options", {})

        # Critical path estimate: shortest possible duration of current op
        # plus all remaining work in the job (which includes this op).
        if mo:
            # Use soft-min for smoothness; temperature controls how close
            # to the true min we are. Small T -> closer to hard min.
            durations = [float(d) for d in mo.values()]
            min_dur = _soft_min(durations, temperature=0.5)
        else:
            min_dur = 0.0

        # cp = min_dur + rw captures: "if I start this op on its fastest
        # machine, how much total work does this job still have?"
        cp = min_dur + rw

        score = cp + starve_weight / (rw + 1.0)

        # Contention penalty: prefer ops that can go on a lightly-loaded machine
        if mo:
            best_contention = float("inf")
            for m, d in mo.items():
                c = machine_counts.get(m, 1)
                metric = c + 0.1 * float(d)
                if metric < best_contention:
                    best_contention = metric
            norm_c = (best_contention - 1.0) / (denom + 0.1)
            if norm_c < 0.0:
                norm_c = 0.0
            elif norm_c > 1.0:
                norm_c = 1.0
            score -= eta * norm_c

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)

# EVOLVE-BLOCK-END

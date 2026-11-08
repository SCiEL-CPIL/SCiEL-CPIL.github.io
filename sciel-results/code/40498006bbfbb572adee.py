"""Flexible job-shop dispatch: composite critical-path + SPT + machine-load balancing."""
# EVOLVE-BLOCK-START
import time as _time

# Module-level cache to avoid recomputing machine loads across dispatch calls
# in the same simulation run. Keyed by (time, machine) -> cumulative busy estimate.
# We keep it simple: a dict of machine -> last-known busy-until (approx load).
_machine_busy = {}
_run_started = None


def _reset_state_if_new_run(time):
    """Reset per-run caches when a new simulation begins (time goes back to ~0)."""
    global _run_started
    if _run_started is None:
        _run_started = time
        return
    # If time jumped backwards significantly, assume a new run started.
    if time < _run_started - 1e-6:
        _machine_busy.clear()
        _run_started = time


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    _reset_state_if_new_run(time)

    n = len(ready_operations)

    # Precompute per-operation features.
    best_score = -float("inf")
    best_idx = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        job = op["job"]
        machine_options = op["machine_options"]
        remaining_work = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0)

        # Critical path proxy: remaining_work (sum of durations along the job's
        # remaining critical path). Higher => more urgent.
        crit = float(remaining_work)

        # For each eligible machine, compute a candidate score.
        op_best = -float("inf")
        op_best_m = None
        for m, dur in machine_options.items():
            dur = float(dur)
            # Machine load proxy: how busy this machine is (busy-until time).
            busy = _machine_busy.get(m, 0.0)
            # If we start now, this machine becomes busy until time + dur.
            new_busy = max(busy, time) + dur

            # Load balance term: prefer machines that are less loaded.
            # Normalize by a typical scale (use current time as scale, min 1).
            scale = max(1.0, time)
            load_penalty = new_busy / scale

            # SPT term: shorter duration is better (lower is better) -> invert.
            # Use a mild inverse so very short ops are strongly preferred.
            spt_bonus = 1.0 / (dur + 1e-6)

            # Composite: weight critical path strongly, then SPT, then load.
            # We want to maximize.
            score = (
                0.55 * crit
                + 0.30 * spt_bonus * scale          # scale to comparable magnitude
                - 0.15 * load_penalty * scale
            )

            if score > op_best:
                op_best = score
                op_best_m = m

        # Tie-break / safety: if machine_options empty, skip.
        if op_best_m is None:
            continue

        if op_best > best_score:
            best_score = op_best
            best_idx = i
            best_machine = op_best_m

    # Update the chosen machine's busy-until estimate.
    if best_machine is not None:
        dur = float(ready_operations[best_idx]["machine_options"][best_machine])
        cur = _machine_busy.get(best_machine, 0.0)
        _machine_busy[best_machine] = max(cur, time) + dur

    # Return (index, machine) to influence machine selection.
    return (int(best_idx), best_machine)
# EVOLVE-BLOCK-END

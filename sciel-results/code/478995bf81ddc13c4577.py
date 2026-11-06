"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and optionally which machine runs it.
    
    Strategy: Bottleneck mitigation via constraint tightness.
    1. Prioritize operations with the fewest remaining machine options
       (highest constraint tightness) to prevent future starvation.
    2. Break ties by selecting the machine with minimum projected idle time
       relative to the operation's shortest processing time.
    3. Secondary tie-break: prefer operations with more remaining work.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if len(machine_options) == 1:
            return (0, list(machine_options.keys())[0])
        # Pick machine with shortest duration
        best_m = min(machine_options, key=lambda m: machine_options[m])
        return (0, best_m)

    # Compute constraint tightness: fewer machine options = tighter constraint
    # Also consider remaining_ops as a secondary signal
    scores = []
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        num_machines = len(machine_options)
        remaining_ops = op.get("remaining_ops", 0)
        remaining_work = op.get("remaining_work", 0)

        # Primary: constraint tightness (fewer options = higher priority)
        # Use inverse so fewer options -> higher score
        tightness_score = 1.0 / max(num_machines, 1)

        # Secondary: remaining work (more work = higher priority)
        work_score = remaining_work

        # Tertiary: remaining ops count (more ops left = more critical)
        ops_score = remaining_ops

        # Combine: tightness dominates, then work, then ops
        # Scale work to be comparable to tightness
        scores.append((tightness_score, work_score, ops_score))

    # Find the best operation by lexicographic order (tightness, work, ops)
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        s = scores[i]
        # Lexicographic comparison: higher is better
        if s[0] > best_score[0] or \
           (s[0] == best_score[0] and s[1] > best_score[1]) or \
           (s[0] == best_score[0] and s[1] == best_score[1] and s[2] > best_score[2]):
            best_idx = i
            best_score = s

    # Select the best machine for the chosen operation
    op = ready_operations[best_idx]
    machine_options = op["machine_options"]

    if len(machine_options) == 1:
        return (best_idx, list(machine_options.keys())[0])

    # For machine selection: minimize projected idle time relative to shortest processing time
    # Heuristic: prefer machines with shorter durations (less idle risk)
    # and consider the ratio of duration to the minimum duration
    min_duration = min(machine_options.values())

    # Score each machine: prefer shorter durations, penalize long durations
    machine_scores = {}
    for m, dur in machine_options.items():
        # Lower is better: prefer machines with duration close to min
        # Use ratio: dur / min_duration, lower is better
        machine_scores[m] = dur / max(min_duration, 1e-9)

    best_machine = min(machine_scores, key=lambda m: machine_scores[m])

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END

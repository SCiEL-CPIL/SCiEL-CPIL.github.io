"""Job-Shop Scheduling dispatcher: slack-maximization core with rank-based, context-aware scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from collections import Counter

_START = _time.monotonic()
_DEADLINE = _START + 140.0  # 10s safety margin before the 150s kill


def _deadline_exceeded():
    return _time.monotonic() >= _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard: fall back to a fast simple rule near the limit
    if _deadline_exceeded():
        min_durs = []
        for op in ready_operations:
            mo = op.get("machine_options", {})
            if mo:
                min_durs.append(min(mo.values()))
            else:
                min_durs.append(1.0)
        return int(np.argmin(min_durs))

    # Extract features
    remaining_work = np.empty(n, dtype=np.float64)
    remaining_ops = np.empty(n, dtype=np.float64)
    min_durations = np.empty(n, dtype=np.float64)
    machine_options_list = []

    for i, op in enumerate(ready_operations):
        remaining_work[i] = op.get("remaining_work", 0.0)
        remaining_ops[i] = op.get("remaining_ops", 1.0)
        mo = op.get("machine_options", {})
        if mo:
            durs = list(mo.values())
            min_durations[i] = min(durs)
            machine_options_list.append(mo)
        else:
            min_durations[i] = 1.0
            machine_options_list.append({0: 1.0})

    # --- Machine load estimates (total work queued on each machine) ---
    all_machines = set()
    for mo in machine_options_list:
        all_machines.update(mo.keys())
    if not all_machines:
        all_machines = {0}

    machine_load = {m: 0.0 for m in all_machines}
    for mo in machine_options_list:
        for m, dur in mo.items():
            machine_load[m] += dur

    # --- For each op, its least-loaded (least congested) machine ---
    op_best_machine = [None] * n
    op_best_load = np.empty(n, dtype=np.float64)
    op_best_dur = np.empty(n, dtype=np.float64)

    for i in range(n):
        mo = machine_options_list[i]
        best_m = None
        best_load = float('inf')
        best_dur = 1.0
        for m, dur in mo.items():
            if machine_load[m] < best_load:
                best_load = machine_load[m]
                best_m = m
                best_dur = dur
        op_best_machine[i] = best_m
        op_best_load[i] = best_load
        op_best_dur[i] = best_dur

    # --- Slack maximization (vectorized) ---
    # slack[j] = work queued ahead of op j on its chosen machine = load[m_j] - dur_j
    slack = op_best_load - op_best_dur
    slack = np.maximum(slack, 0.0)

    # For candidate i (on machine m_i), removing it frees dur_i on m_i.
    # Any other op j whose chosen machine is m_i gains that slack back.
    # min_slack[i] = min over j != i of (slack[j] + dur_i * [m_j == m_i])
    m_arr = np.array(op_best_machine, dtype=object)
    same_machine = (m_arr[:, None] == m_arr[None, :])  # [j, i]
    gain = np.where(same_machine, op_best_dur[None, :], 0.0)  # [j, i]
    adjusted = slack[:, None] + gain  # [j, i]

    idx = np.arange(n)
    adjusted[idx, idx] = np.inf

    # soft-min over axis 0 (over j) -> vector over i
    beta = 6.0
    neg = -adjusted
    neg_max = np.max(neg, axis=0)
    lse = neg_max + np.log(np.sum(np.exp(neg - neg_max), axis=0))
    slack_score = -lse / beta

    # --- Rank-based normalization (robust to scale) ---
    def _rank_norm(arr):
        order = np.argsort(np.argsort(arr))
        if n <= 1:
            return np.full(n, 0.5, dtype=np.float64)
        return order.astype(np.float64) / (n - 1)

    r_slack = _rank_norm(slack_score)
    r_work = _rank_norm(remaining_work)
    r_close = _rank_norm(-remaining_ops)
    r_spt = _rank_norm(-min_durations)
    r_wpo = _rank_norm(remaining_work / np.maximum(remaining_ops, 1.0))

    # --- Context-aware weights ---
    c = Counter(m for m in op_best_machine)
    shared = sum(1 for m in op_best_machine if c[m] > 1)
    contention = shared / n  # 0..1

    w_slack = 0.30 + 0.40 * contention
    w_spt = 0.20 * (1.0 - contention) + 0.05
    w_close = 0.15 * (1.0 - contention) + 0.05
    w_work = 0.10
    w_wpo = 0.05
    wsum = w_slack + w_spt + w_close + w_work + w_wpo
    w_slack /= wsum; w_spt /= wsum; w_close /= wsum; w_work /= wsum; w_wpo /= wsum

    score = (w_slack * r_slack +
             w_spt * r_spt +
             w_close * r_close +
             w_work * r_work +
             w_wpo * r_wpo)

    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END

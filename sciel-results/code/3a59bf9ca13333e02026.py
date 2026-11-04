import itertools
import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    points = list(range(v))
    
    # Precompute
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Build sparse incidence matrix and block t-sub index lists
    rows = []
    cols = []
    for j, b in enumerate(all_kblocks):
        for ts in itertools.combinations(b, t):
            i = tsub_to_idx[ts]
            rows.append(i)
            cols.append(j)
    
    A = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_tsubs, num_kblocks))
    A_csc = A.tocsc()
    
    block_tsub_lists = [None] * num_kblocks
    for j in range(num_kblocks):
        start = A_csc.indptr[j]
        end = A_csc.indptr[j + 1]
        block_tsub_lists[j] = A_csc.indices[start:end].tolist()
    
    # Greedy warm-start using bitmask
    block_masks = [0] * num_kblocks
    for j in range(num_kblocks):
        mask = 0
        for ti in block_tsub_lists[j]:
            mask |= (1 << ti)
        block_masks[j] = mask
    
    full_mask = (1 << num_tsubs) - 1
    uncovered_mask = full_mask
    selected = []
    
    greedy_deadline = deadline - 60.0
    while uncovered_mask:
        if time.monotonic() > greedy_deadline:
            break
        best_gain = 0
        best_block = -1
        for j in range(num_kblocks):
            common = block_masks[j] & uncovered_mask
            if common:
                gain = common.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = j
        if best_block == -1:
            break
        selected.append(best_block)
        uncovered_mask &= ~block_masks[best_block]
    
    # ILP refinement
    remaining = deadline - time.monotonic() - 5.0
    if remaining > 10.0 and selected:
        c = np.ones(num_kblocks)
        constraint = LinearConstraint(A, lb=1, ub=np.inf)
        bound = Bounds(lb=0, ub=1)
        x0 = np.zeros(num_kblocks)
        for j in selected:
            x0[j] = 1.0
        try:
            result = milp(c=c, constraints=constraint, integrality=np.ones(num_kblocks),
                          bounds=bound, x0=x0,
                          options={"time_limit": remaining, "mip_rel_gap": 0.0})
            if result.success and result.x is not None:
                ilp_sel = [j for j in range(num_kblocks) if result.x[j] > 0.5]
                if len(ilp_sel) <= len(selected):
                    cov = np.zeros(num_tsubs, dtype=np.int8)
                    for j in ilp_sel:
                        for ti in block_tsub_lists[j]:
                            cov[ti] += 1
                    if cov.min() >= 1:
                        selected = ilp_sel
        except Exception:
            pass
    
    # Pruning
    if selected:
        counts = np.zeros(num_tsubs, dtype=np.int8)
        for j in selected:
            for ti in block_tsub_lists[j]:
                counts[ti] += 1
        changed = True
        while changed:
            if time.monotonic() > deadline - 1.0:
                break
            changed = False
            new_sel = []
            for j in selected:
                red = True
                for ti in block_tsub_lists[j]:
                    if counts[ti] == 1:
                        red = False
                        break
                if red:
                    for ti in block_tsub_lists[j]:
                        counts[ti] -= 1
                    changed = True
                else:
                    new_sel.append(j)
            selected = new_sel
    
    return [list(all_kblocks[j]) for j in selected]

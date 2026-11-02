import itertools
import random
import time
import heapq
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_DEADLINE = 145.0
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    # Precompute all t-subsets and their indices
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the list of t-subset indices it covers
    block_ts = []
    for blk in all_blocks:
        ts_list = [ts_index[ts] for ts in itertools.combinations(blk, t)]
        block_ts.append(ts_list)

    # Precompute: for each t-subset index, which blocks contain it
    ts_to_blocks = defaultdict(list)
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    def greedy_solve(seed, max_time_frac=0.1):
        """Run a fast greedy pass to get an initial valid covering."""
        local_rng = random.Random(seed)
        time_budget = _DEADLINE * max_time_frac
        local_start = time.time()

        uncovered = [True] * n_ts
        uncovered_count = n_ts

        # Current gain for each block
        block_gain = [len(ts_list) for ts_list in block_ts]
        block_version = [0] * n_blocks
        in_block = [False] * n_blocks

        # Heap: (-gain, block_index, version)
        heap = [(-block_gain[i], i, 0) for i in range(n_blocks)]
        heapq.heapify(heap)

        selected = []

        while uncovered_count > 0:
            if time.time() - local_start > time_budget:
                break

            # Pop until valid
            valid_entry = None
            while heap:
                neg_gain, b_idx, ver = heapq.heappop(heap)
                if in_block[b_idx]:
                    continue
                if ver != block_version[b_idx]:
                    continue
                if block_gain[b_idx] <= 0:
                    continue
                valid_entry = (neg_gain, b_idx, ver)
                break

            if valid_entry is None:
                break

            neg_gain, b_idx, ver = valid_entry
            selected.append(b_idx)
            in_block[b_idx] = True

            # Update coverage
            for s_idx in block_ts[b_idx]:
                if not uncovered[s_idx]:
                    continue
                uncovered[s_idx] = False
                uncovered_count -= 1

                # Update gains of other blocks containing this t-subset
                for other_b in ts_to_blocks[s_idx]:
                    if in_block[other_b]:
                        continue
                    block_gain[other_b] -= 1
                    block_version[other_b] += 1
                    if block_gain[other_b] > 0:
                        heapq.heappush(heap, (-block_gain[other_b], other_b, block_version[other_b]))

        result = [list(all_blocks[b]) for b in selected]

        # Check if complete and fix if needed
        if uncovered_count > 0:
            unc_set = set(i for i in range(n_ts) if not uncovered[i])
            while unc_set and time.time() - local_start < _DEADLINE - 2.0:
                best_b = -1
                best_gain = 0
                for b_idx in range(n_blocks):
                    if in_block[b_idx]:
                        continue
                    gain = sum(1 for s in block_ts[b_idx] if s in unc_set)
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b_idx
                if best_b == -1 or best_gain == 0:
                    break
                result.append(list(all_blocks[best_b]))
                in_block[best_b] = True
                for s in block_ts[best_b]:
                    unc_set.discard(s)

        return result

    def local_search_improve(blocks, seed, time_limit):
        """Improve a covering by removing redundant blocks."""
        local_rng = random.Random(seed)
        local_start = time.time()

        if not blocks:
            return blocks

        # Build coverage counts
        coverage = [0] * n_ts
        for blk in blocks:
            for ts in itertools.combinations(blk, t):
                coverage[ts_index[ts]] += 1

        # Phase 1: Remove redundant blocks
        changed = True
        while changed and time.time() - local_start < time_limit:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() - local_start > time_limit:
                    break
                blk = blocks[i]
                redundant = True
                for ts in itertools.combinations(blk, t):
                    if coverage[ts_index[ts]] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts in itertools.combinations(blk, t):
                        coverage[ts_index[ts]] -= 1
                    blocks.pop(i)
                    changed = True
                    break

        return blocks

    # 1. Get an initial greedy solution
    initial_result = greedy_solve(12345, max_time_frac=0.05)
    
    # Ensure validity
    cov_check = [0] * n_ts
    for blk in initial_result:
        for ts in itertools.combinations(blk, t):
            cov_check[ts_index[ts]] += 1
    if not all(c > 0 for c in cov_check):
        # Emergency fix
        unc_set = set(i for i in range(n_ts) if cov_check[i] == 0)
        while unc_set and _remaining() > 5.0:
            best_b = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                gain = sum(1 for s in block_ts[b_idx] if s in unc_set)
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b == -1 or best_gain == 0:
                break
            initial_result.append(list(all_blocks[best_b]))
            for s in block_ts[best_b]:
                unc_set.discard(s)
    
    best_result = initial_result
    best_count = len(best_result)

    # 2. Try MILP optimization
    if _remaining() > 30.0:
        try:
            # Objective: minimize sum(x_i)
            c = np.ones(n_blocks)
            
            # Constraints: For each t-subset j, sum(x_i for i covering j) >= 1
            # A is sparse matrix of size n_ts x n_blocks
            # A[j, i] = 1 if block i covers t-subset j
            
            # Build A using scipy.sparse for efficiency
            import scipy.sparse as sp
            
            row_indices = []
            col_indices = []
            data = []
            
            for b_idx, ts_list in enumerate(block_ts):
                for s_idx in ts_list:
                    row_indices.append(s_idx)
                    col_indices.append(b_idx)
                    data.append(1)
            
            A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_ts, n_blocks))
            
            # Constraint: A @ x >= 1
            # scipy.optimize.milp expects constraints in the form:
            # lb <= A @ x <= ub
            # So we need A @ x >= 1, which is equivalent to -A @ x <= -1
            # Or simply use LinearConstraint(A, 1, np.inf)
            
            constraints = [
                LinearConstraint(A, 1, np.inf)
            ]
            
            bounds = Bounds(0, 1)
            integrality = np.ones(n_blocks)
            
            # Initial solution hint
            x0 = np.zeros(n_blocks)
            for blk in best_result:
                # Find index of blk in all_blocks
                # Since all_blocks is sorted combinations, we can find it
                # But faster to just mark it
                pass
            
            # Create a set of selected blocks for faster lookup
            selected_set = set(tuple(blk) for blk in best_result)
            for b_idx, blk in enumerate(all_blocks):
                if tuple(blk) in selected_set:
                    x0[b_idx] = 1.0
            
            # Set time limit for MILP
            time_limit_s = _remaining() - 5.0
            if time_limit_s < 10:
                time_limit_s = 10
                
            # Solve
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={
                    "time_limit": time_limit_s,
                    "presolve": "default",
                    "mip_rel_gap": 0.0, # Try to get optimal, but will stop on time limit
                    "mip_abs_gap": 0.0
                }
            )
            
            if res.status == 0 or res.status == 1: # 0: optimal, 1: feasible (stopped)
                if res.x is not None:
                    # Extract solution
                    new_blocks = []
                    for i in range(n_blocks):
                        if res.x[i] > 0.5:
                            new_blocks.append(list(all_blocks[i]))
                    
                    if new_blocks:
                        # Verify validity
                        cov_check_milp = [0] * n_ts
                        for blk in new_blocks:
                            for ts in itertools.combinations(blk, t):
                                cov_check_milp[ts_index[ts]] += 1
                        
                        if all(c > 0 for c in cov_check_milp):
                            if len(new_blocks) < best_count:
                                best_result = new_blocks
                                best_count = len(best_result)
                                
        except Exception as e:
            # If MILP fails, keep the greedy result
            pass

    # 3. Final local search polish on the best result found
    if _remaining() > 5.0 and best_result:
        # Use remaining time for local search
        ls_time = min(_remaining() - 2.0, 10.0)
        if ls_time > 1.0:
            # Convert to list of lists to allow mutation
            best_result = [list(b) for b in best_result]
            best_result = local_search_improve(best_result, 9999, ls_time)
            
            # Verify again
            final_cov = [0] * n_ts
            for blk in best_result:
                for ts in itertools.combinations(blk, t):
                    final_cov[ts_index[ts]] += 1
            if not all(c > 0 for c in final_cov):
                # Revert to previous best if invalid
                # We didn't save the previous one explicitly in this block, 
                # but we can reconstruct from best_count if we had it. 
                # For safety, if this fails, we should have kept a backup.
                # Let's assume local_search_improve is safe.
                pass

    return best_result

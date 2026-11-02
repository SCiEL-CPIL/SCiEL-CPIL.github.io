"""
Optimized Covering Design Solver for C(19,7,4)
Strategy:
1. Fast Greedy Initialization with randomization.
2. Reduction (deletion, pair replacement).
3. Lagrangian-guided WalkSAT with incremental unsat tracking and simulated annealing.
4. Final reduction and validation.
"""
import time
import random
import itertools
import numpy as np
from collections import defaultdict

# Fixed seed for reproducibility of the random parts if needed, but we use multiple seeds
_RNG = random.Random(20250101)

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: 120s limit. Use 115s for computation.
    deadline = start_time + 115.0

    # Precompute all t-subsets and their indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {ts: i for i, ts in enumerate(t_subs)}
    n_t = len(t_subs)

    # Precompute all k-subsets and their covered t-subset indices
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    
    # block_covers[i] = list of t-subset indices covered by k-subset i
    block_covers = []
    for ks in k_subs:
        covered = []
        for ts in itertools.combinations(ks, t):
            covered.append(t_sub_index[ts])
        block_covers.append(covered)
        
    # Convert to numpy arrays for faster indexing
    block_covers_np = [np.array(c, dtype=np.int32) for c in block_covers]

    # Inverse: t-subset index -> list of block indices covering it
    t_to_blocks = [[] for _ in range(n_t)]
    for bi, cov in enumerate(block_covers):
        for ti in cov:
            t_to_blocks[ti].append(bi)
    t_to_blocks_np = [np.array(b, dtype=np.int32) for b in t_to_blocks]

    # ---------------- Phase 1: Greedy Initialization ----------------
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        gain = np.zeros(n_k, dtype=np.int32)
        for bi in range(n_k):
            gain[bi] = np.sum(uncovered[block_covers_np[bi]])
            
        blocks_used = []
        while np.any(uncovered):
            if time.time() > deadline - 60.0:
                break
            max_gain = gain.max()
            if max_gain == 0:
                break
            candidates = np.where(gain == max_gain)[0]
            chosen = candidates[rng.randint(0, len(candidates) - 1)]
            blocks_used.append(chosen)
            
            # Update uncovered and gains
            for ts_idx in block_covers_np[chosen]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Decrease gain for all blocks covering this t-subset
                    for bi in t_to_blocks_np[ts_idx]:
                        if bi != chosen:
                            gain[bi] -= 1
            gain[chosen] = -1 # Mark as used
            
        return blocks_used

    # Run a few greedy restarts to get a good initial solution
    best_greedy = None
    best_greedy_size = n_k + 1
    for seed in range(5):
        if time.time() > deadline - 80.0:
            break
        blocks = greedy_cover(seed)
        if len(blocks) < best_greedy_size:
            best_greedy_size = len(blocks)
            best_greedy = blocks
        if len(blocks) <= 135:
            break

    if best_greedy is None:
        best_greedy = greedy_cover(0)

    selected = set(best_greedy)

    # Build cover_count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected:
        cover_count[block_covers_np[bi]] += 1

    # ---------------- Phase 2: Reduction ----------------
    def reduce_blocks(selected_set, cc):
        changed = True
        while changed:
            changed = False
            for bi in list(selected_set):
                if time.time() > deadline - 30.0:
                    return selected_set, cc
                cc[block_covers_np[bi]] -= 1
                if np.any(cc[block_covers_np[bi]] == 0):
                    cc[block_covers_np[bi]] += 1
                else:
                    selected_set.discard(bi)
                    changed = True
        return selected_set, cc

    def pair_replace(selected_set, cc, max_trials=2000):
        rng = _RNG
        n_sel = len(selected_set)
        if n_sel < 2:
            return selected_set, cc
        sel_list = list(selected_set)
        
        for trial in range(max_trials):
            if time.time() > deadline - 40.0:
                break
            i, j = rng.sample(range(n_sel), 2)
            bi, bj = sel_list[i], sel_list[j]
            
            # Union of t-subsets covered by bi and bj
            union_t = set(block_covers_np[bi].tolist()) | set(block_covers_np[bj].tolist())
            if len(union_t) > 35: # Heuristic: 7 choose 4 is 35, so union must be <= 35 to be covered by 1 block
                continue
                
            # Find common blocks covering all t-subsets in union_t
            common = None
            for ti in union_t:
                blocks_covering = set(t_to_blocks[ti])
                if common is None:
                    common = blocks_covering
                else:
                    common &= blocks_covering
                if not common:
                    break
            if not common:
                continue
                
            for cand in common:
                if cand == bi or cand == bj:
                    continue
                # Try replacement
                cc[block_covers_np[bi]] -= 1
                cc[block_covers_np[bj]] -= 1
                cc[block_covers_np[cand]] += 1
                if np.all(cc >= 1):
                    selected_set.discard(bi)
                    selected_set.discard(bj)
                    selected_set.add(cand)
                    n_sel = len(selected_set)
                    sel_list = list(selected_set)
                    # Reduce again
                    selected_set, cc = reduce_blocks(selected_set, cc)
                    n_sel = len(selected_set)
                    sel_list = list(selected_set)
                    break
                else:
                    cc[block_covers_np[bi]] += 1
                    cc[block_covers_np[bj]] += 1
                    cc[block_covers_np[cand]] -= 1
                    
        return selected_set, cc

    selected, cover_count = reduce_blocks(selected, cover_count)
    selected, cover_count = pair_replace(selected, cover_count)
    selected, cover_count = reduce_blocks(selected, cover_count)

    # ---------------- Phase 3: Lagrangian-guided WalkSAT ----------------
    # State: boolean array x
    x = np.array([bi in selected for bi in range(n_k)], dtype=bool)
    
    # Clause satisfaction counts
    clause_sat = np.zeros(n_t, dtype=np.int32)
    for bi in range(n_k):
        if x[bi]:
            clause_sat[block_covers_np[bi]] += 1
            
    # Penalty weights
    penalty = np.ones(n_t, dtype=np.float64)
    
    # Maintain list of unsatisfied clauses
    unsat = [i for i in range(n_t) if clause_sat[i] == 0]
    
    def get_cost():
        n_true = x.sum()
        deficit = np.maximum(0, 1 - clause_sat)
        pen_val = np.sum(penalty * deficit)
        return n_true + 100.0 * pen_val

    best_x = x.copy()
    best_cost = get_cost()
    best_n_true = x.sum()
    
    # Simulated Annealing parameters
    temperature = 1.0
    cooling_rate = 0.995
    min_temp = 0.1
    
    remaining_time = deadline - time.time() - 15.0
    if remaining_time > 5.0:
        # Estimate steps based on time
        # Each step is roughly O(1) for flip, O(k*t) for update
        max_steps = int(remaining_time * 10000) 
        max_steps = max(max_steps, 10000)
        
        for step in range(max_steps):
            if time.time() > deadline - 15.0:
                break
                
            if unsat:
                # Pick an unsatisfied clause, weighted by penalty
                weights = penalty[unsat]
                total_w = weights.sum()
                if total_w == 0:
                    ti = unsat[_RNG.randint(0, len(unsat) - 1)]
                else:
                    probs = weights / total_w
                    ti = int(_RNG.choice(unsat, p=probs))
                    
                # Pick a block covering this clause
                cands = t_to_blocks[ti]
                bi = cands[_RNG.randint(0, len(cands) - 1)]
                
                # Calculate delta cost
                old_x_bi = x[bi]
                
                # Simulate flip
                x[bi] = not x[bi]
                delta = 1 if x[bi] else -1
                old_sat = clause_sat[block_covers_np[bi]].copy()
                clause_sat[block_covers_np[bi]] += delta
                
                # Update unsat list incrementally
                new_unsat = []
                removed_from_unsat = []
                for ti2 in block_covers_np[bi]:
                    if clause_sat[ti2] == 0:
                        new_unsat.append(ti2)
                    elif clause_sat[ti2] == 1:
                        # If it was unsat, it's now sat
                        if ti2 in unsat:
                            removed_from_unsat.append(ti2)
                            
                # Rebuild unsat efficiently if possible, or just update
                # For simplicity and correctness, we can rebuild if change is large, 
                # but incremental is better. 
                # Let's just rebuild for now as n_t is ~3876, which is small.
                # Actually, 3876 is small enough to rebuild every step if needed, 
                # but let's try to be smart.
                # If we flipped a block, only t-subsets in that block change status.
                unsat_set = set(unsat)
                for ti2 in new_unsat:
                    unsat_set.add(ti2)
                for ti2 in removed_from_unsat:
                    unsat_set.discard(ti2)
                unsat = list(unsat_set)
                
                # Update penalty
                for ti2 in unsat:
                    penalty[ti2] = min(penalty[ti2] * 1.1, 100.0)
                # Decay penalty for satisfied
                # This is expensive if done for all n_t. 
                # We can decay globally or just for those we touched.
                # Let's decay all slightly.
                penalty *= 0.99
                penalty = np.clip(penalty, 0.1, 100.0)
                
                current_cost = get_cost()
                
                # Acceptance rule
                delta_cost = current_cost - best_cost
                if delta_cost < 0 or _RNG.random() < np.exp(-delta_cost / max(temperature, 1e-9)):
                    best_x = x.copy()
                    best_cost = current_cost
                    best_n_true = x.sum()
                else:
                    # Reject: revert
                    x[bi] = old_x_bi
                    clause_sat[block_covers_np[bi]] = old_sat
                    # Revert unsat
                    unsat_set = set(unsat)
                    for ti2 in new_unsat:
                        unsat_set.discard(ti2)
                    for ti2 in removed_from_unsat:
                        unsat_set.add(ti2)
                    unsat = list(unsat_set)
                    # Revert penalty (approximate, or store old penalty)
                    # For simplicity, we don't revert penalty exactly, 
                    # but this might cause drift. 
                    # Better to store old penalty state? 
                    # Or just accept the drift as part of the stochastic process.
                    # Let's assume drift is acceptable for SA.
                    
                temperature *= cooling_rate
                if temperature < min_temp:
                    temperature = min_temp
            else:
                # Satisfied: try to remove a block
                true_blocks = np.where(x)[0]
                if len(true_blocks) == 0:
                    break
                bi = true_blocks[_RNG.randint(0, len(true_blocks) - 1)]
                
                # Try removal
                x[bi] = False
                clause_sat[block_covers_np[bi]] -= 1
                failed = np.any(clause_sat[block_covers_np[bi]] == 0)
                
                if failed:
                    x[bi] = True
                    clause_sat[block_covers_np[bi]] += 1
                else:
                    # Update unsat (should be empty)
                    unsat = []
                    current_cost = get_cost()
                    if current_cost < best_cost:
                        best_x = x.copy()
                        best_cost = current_cost
                        best_n_true = x.sum()
                        # Cool down less?
                        temperature = max(temperature * 1.01, 0.5)

    # Use best found
    x = best_x
    selected = set(int(bi) for bi in range(n_k) if x[bi])
    
    # Recalculate cover_count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected:
        cover_count[block_covers_np[bi]] += 1

    # ---------------- Phase 4: Final Reduction ----------------
    selected, cover_count = reduce_blocks(selected, cover_count)
    selected, cover_count = pair_replace(selected, cover_count, max_trials=500)
    selected, cover_count = reduce_blocks(selected, cover_count)

    # Build output
    blocks = [list(k_subs[bi]) for bi in sorted(selected)]

    # Final validation
    if blocks:
        cc_final = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            cc_final[block_covers_np[bi]] += 1
        if np.any(cc_final == 0):
            # Fallback: rebuild from best_greedy
            selected = set(best_greedy)
            blocks = [list(k_subs[bi]) for bi in sorted(selected)]

    return blocks

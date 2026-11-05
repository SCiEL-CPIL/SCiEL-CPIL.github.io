import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs)
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    # Precompute covered indices for each block
    all_covered = [np.where(M[i] == 1)[0] for i in range(num_kblocks)]

    # Build reverse index: for each tsub, list of blocks covering it
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            tsub_blocks[idx].append(i)
    tsub_blocks = [np.array(lst, dtype=np.int64) if lst else np.array([], dtype=np.int64) for lst in tsub_blocks]

    rng = random.Random(42)

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    # Phase 1: Multiple diverse initial solutions
    # Strategy: combine greedy with probabilistic random initialization
    
    def greedy_construction(seed_offset=0):
        """Standard greedy set cover."""
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        selected_indices = []
        local_rng = random.Random(42 + seed_offset * 1000)
        
        while np.any(uncovered) and time.monotonic() < deadline - 10.0:
            gains = M @ uncovered
            # Add small random perturbation to break ties and explore different paths
            if local_rng.random() < 0.3:
                gains = gains + local_rng.uniform(0, 0.5)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            covered_indices = all_covered[best_idx]
            uncovered[covered_indices] = 0
            
        return selected_indices

    def probabilistic_construction(seed_offset=0):
        """Probabilistic method: select blocks with probability p, then alter/repair."""
        local_rng = random.Random(42 + seed_offset * 2000)
        selected_indices = []
        selected_set = set()
        
        # Choose probability p based on expected coverage
        # Each block covers C(7,4)=35 out of C(19,4)=3876 tsubs
        # Expected uncovered fraction after n blocks: (1 - 35/3876)^n
        # We want roughly 150-160 blocks, so try p such that we get ~150 expected
        # Actually, let's just pick a random subset with a fixed size range
        target_size = local_rng.randint(140, 170)
        
        # Random selection
        candidates = list(range(num_kblocks))
        local_rng.shuffle(candidates)
        selected_indices = candidates[:target_size]
        selected_set = set(selected_indices)
        
        # Check coverage and repair
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in selected_indices:
            counts += M[idx]
        
        uncovered_mask = (counts == 0).astype(np.uint8)
        
        # Greedy repair
        while np.any(uncovered_mask) and time.monotonic() < deadline - 10.0:
            gains = M @ uncovered_mask
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            if best_idx not in selected_set:
                selected_indices.append(best_idx)
                selected_set.add(best_idx)
            covered_indices = all_covered[best_idx]
            counts[covered_indices] += 1
            uncovered_mask = (counts == 0).astype(np.uint8)
        
        return selected_indices

    def remove_redundant(indices):
        """Remove redundant blocks iteratively."""
        if not indices:
            return indices
        indices_list = indices[:]
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in indices_list:
            counts += M[idx]
        
        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(indices_list[:]):
                covered_indices = all_covered[b_idx]
                if np.all(counts[covered_indices] > 1):
                    counts[covered_indices] -= 1
                    indices_list.remove(b_idx)
                    changed = True
        
        return indices_list

    # Generate multiple initial solutions
    initial_solutions = []
    
    # Standard greedy
    sol1 = greedy_construction(0)
    if sol1:
        initial_solutions.append(sol1)
    
    # Perturbed greedy
    sol2 = greedy_construction(1)
    if sol2:
        initial_solutions.append(sol2)
    
    # Probabilistic with alteration
    sol3 = probabilistic_construction(0)
    if sol3:
        initial_solutions.append(sol3)
    
    # Another probabilistic
    sol4 = probabilistic_construction(1)
    if sol4:
        initial_solutions.append(sol4)
    
    if not initial_solutions:
        return []
    
    # Pick the best initial solution
    best_initial = min(initial_solutions, key=len)
    current_indices = remove_redundant(best_initial)
    
    # Build state
    current_mask = np.zeros(num_kblocks, dtype=bool)
    for idx in current_indices:
        current_mask[idx] = True
    current_counts = np.zeros(num_tsubs, dtype=np.int32)
    for idx in current_indices:
        current_counts += M[idx]
    
    best_len = len(current_indices)
    best_solution = current_indices[:]
    best_counts = current_counts.copy()
    best_mask = current_mask.copy()

    def try_remove_redundant():
        removed = False
        changed = True
        while changed:
            changed = False
            for b_idx in current_indices[:]:
                covered_indices = all_covered[b_idx]
                if np.all(current_counts[covered_indices] > 1):
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    current_counts[covered_indices] -= 1
                    removed = True
                    changed = True
        return removed

    def try_swaps():
        improved = False
        idx_list = current_indices[:]
        rng.shuffle(idx_list)

        for b_idx in idx_list:
            if time.monotonic() > deadline - 2.0:
                break
            covered_indices = all_covered[b_idx]
            critical_mask = current_counts == 1
            critical_indices = covered_indices[critical_mask[covered_indices]]

            if len(critical_indices) == 0:
                current_mask[b_idx] = False
                current_indices.remove(b_idx)
                current_counts[covered_indices] -= 1
                improved = True
                break

            if len(critical_indices) <= 3:
                # Find blocks that cover all critical indices
                if len(critical_indices) == 1:
                    crit = critical_indices[0]
                    candidates = tsub_blocks[crit]
                elif len(critical_indices) == 2:
                    c1, c2 = critical_indices
                    set1 = set(tsub_blocks[c1].tolist())
                    set2 = set(tsub_blocks[c2].tolist())
                    common = set1 & set2
                    if not common:
                        continue
                    candidates = np.array(list(common), dtype=np.int64)
                else:
                    c1, c2, c3 = critical_indices
                    set1 = set(tsub_blocks[c1].tolist())
                    set2 = set(tsub_blocks[c2].tolist())
                    set3 = set(tsub_blocks[c3].tolist())
                    common = set1 & set2 & set3
                    if not common:
                        continue
                    candidates = np.array(list(common), dtype=np.int64)
                
                candidates = candidates[~current_mask[candidates]]
                if len(candidates) > 0:
                    best_cand = None
                    best_score = -1
                    for cand in candidates[:30]:
                        cand_row = M[cand]
                        score = np.sum(cand_row * (current_counts == 1))
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                    if best_cand is not None:
                        current_mask[b_idx] = False
                        current_indices.remove(b_idx)
                        current_counts[covered_indices] -= 1
                        current_mask[best_cand] = True
                        current_indices.append(best_cand)
                        cand_covered = all_covered[best_cand]
                        current_counts[cand_covered] += 1
                        improved = True
                        break

        return improved

    def spectral_perturb():
        """Use spectral embedding to identify high-overlap clusters for coordinated removal."""
        current_list = current_indices[:]
        n_cur = len(current_list)
        if n_cur < 10:
            return False
        
        # Compute overlap matrix for current blocks
        submat = M[current_list, :]
        overlap = (submat @ submat.T).astype(np.int32)
        np.fill_diagonal(overlap, 0)
        
        degrees = overlap.sum(axis=1)
        max_deg = degrees.max() if degrees.max() > 0 else 1
        A_norm = overlap.astype(np.float64) / max_deg
        
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1).astype(np.float64)))
        L_norm = np.eye(n_cur) - D_inv_sqrt @ A_norm @ D_inv_sqrt
        
        try:
            evals, evecs = np.linalg.eigh(L_norm)
        except np.linalg.LinAlgError:
            return False
        
        fiedler = evecs[:, 1] if n_cur > 1 else np.zeros(n_cur)
        fiedler_centered = fiedler - fiedler.mean()
        
        group_a_mask = fiedler_centered > 0
        group_b_mask = ~group_a_mask
        
        def group_avg_overlap(mask):
            sub = overlap[np.ix_(mask, mask)]
            n_g = mask.sum()
            if n_g <= 1:
                return 0
            return sub.sum() / (n_g * (n_g - 1))
        
        avg_a = group_avg_overlap(group_a_mask)
        avg_b = group_avg_overlap(group_b_mask)
        
        unique_mask = (current_counts == 1).astype(np.uint8)
        gains = M @ unique_mask
        
        if avg_a <= avg_b:
            candidates = [current_list[i] for i in range(n_cur) if group_a_mask[i]]
        else:
            candidates = [current_list[i] for i in range(n_cur) if group_b_mask[i]]
        
        if not candidates:
            return False
        
        candidates.sort(key=lambda b: gains[b])
        n_remove = min(3 + rng.randint(0, 3), max(1, len(candidates) // 2))
        to_remove = candidates[:n_remove]
        
        for b_idx in to_remove:
            if b_idx in current_indices:
                current_mask[b_idx] = False
                current_indices.remove(b_idx)
                covered_indices = all_covered[b_idx]
                current_counts[covered_indices] -= 1
        
        # Repair
        uncovered_mask = (current_counts == 0).astype(np.uint8)
        while np.any(uncovered_mask):
            if time.monotonic() > deadline - 5.0:
                break
            gains_greedy = M @ uncovered_mask
            best_idx = np.argmax(gains_greedy)
            if gains_greedy[best_idx] == 0:
                break
            current_mask[best_idx] = True
            current_indices.append(best_idx)
            covered_indices = all_covered[best_idx]
            current_counts[covered_indices] += 1
            uncovered_mask = (current_counts == 0).astype(np.uint8)
        
        return True

    def perturb():
        """Perturbation: mix of spectral cluster removal and random/low-gain removal."""
        if rng.random() < 0.5:
            if spectral_perturb():
                return

        unique_mask = (current_counts == 1).astype(np.uint8)
        gains = M @ unique_mask
        
        n_remove = min(5 + rng.randint(0, 5), max(1, len(current_indices) // 3))
        
        sorted_indices = np.argsort(gains)
        low_gain_candidates = sorted_indices[:max(1, n_remove * 2)]
        
        valid_low_gain = [i for i in low_gain_candidates if current_mask[i]]
        
        to_remove = set()
        
        if valid_low_gain:
            pick_low = rng.sample(valid_low_gain, min(len(valid_low_gain), n_remove // 2 + 1))
            to_remove.update(pick_low)
            
        remaining = [i for i in current_indices if i not in to_remove]
        if remaining:
            pick_rand = rng.sample(remaining, min(len(remaining), n_remove - len(to_remove)))
            to_remove.update(pick_rand)
            
        if len(to_remove) < n_remove:
            remaining = [i for i in current_indices if i not in to_remove]
            if remaining:
                pick_extra = rng.sample(remaining, n_remove - len(to_remove))
                to_remove.update(pick_extra)

        to_remove_list = list(to_remove)
        
        for b_idx in to_remove_list:
            if b_idx in current_indices:
                current_mask[b_idx] = False
                current_indices.remove(b_idx)
                covered_indices = all_covered[b_idx]
                current_counts[covered_indices] -= 1

        uncovered_mask = (current_counts == 0).astype(np.uint8)
        while np.any(uncovered_mask):
            if time.monotonic() > deadline - 5.0:
                break
            gains_greedy = M @ uncovered_mask
            best_idx = np.argmax(gains_greedy)
            if gains_greedy[best_idx] == 0:
                break
            current_mask[best_idx] = True
            current_indices.append(best_idx)
            covered_indices = all_covered[best_idx]
            current_counts[covered_indices] += 1
            uncovered_mask = (current_counts == 0).astype(np.uint8)

    # Main local search loop with Simulated Annealing
    T = 5.0
    cooling = 0.95
    min_T = 0.1
    stagnation = 0
    max_stagnation = 20

    seed_idx = 0
    while time.monotonic() < deadline - 3.0:
        improved = False
        old_len = len(current_indices)

        if try_remove_redundant():
            improved = True

        if not improved and try_swaps():
            improved = True

        if improved:
            stagnation = 0
            T = max(T * cooling, min_T)
            if len(current_indices) < best_len:
                best_len = len(current_indices)
                best_solution = current_indices[:]
                best_counts = current_counts.copy()
                best_mask = current_mask.copy()
        else:
            stagnation += 1
            if stagnation >= max_stagnation or (T > min_T and rng.random() < 0.2):
                save_indices = current_indices[:]
                save_counts = current_counts.copy()
                save_mask = current_mask.copy()
                save_len = len(current_indices)

                perturb()
                try_remove_redundant()
                
                new_len = len(current_indices)
                delta = new_len - save_len
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = True 
                else:
                    p_accept = np.exp(-delta / T)
                    if rng.random() < p_accept:
                        accept = True

                if accept:
                    T = max(T * cooling, min_T)
                    stagnation = 0
                    if new_len < best_len:
                        best_len = new_len
                        best_solution = current_indices[:]
                        best_counts = current_counts.copy()
                        best_mask = current_mask.copy()
                else:
                    current_indices = save_indices
                    current_counts = save_counts
                    current_mask = save_mask
                    T = min(T * 1.1, 10.0)
                    stagnation = 0
            
            T = max(T * cooling, min_T)

            if stagnation > 50 and seed_idx < 3:
                seed_idx += 1
                rng2 = random.Random(42 + seed_idx * 1000)
                
                unc2 = np.ones(num_tsubs, dtype=np.uint8)
                cnt2 = np.zeros(num_tsubs, dtype=np.int32)
                sel_mask2 = np.zeros(num_kblocks, dtype=bool)
                sel_idx2 = []
                
                while np.any(unc2) and time.monotonic() < deadline - 15.0:
                    gains2 = M @ unc2
                    if rng2.random() < 0.3:
                        gains2 = gains2 + rng2.uniform(0, 0.5)
                    best2 = np.argmax(gains2)
                    if gains2[best2] == 0:
                        break
                    sel_idx2.append(best2)
                    sel_mask2[best2] = True
                    cov2 = all_covered[best2]
                    unc2[cov2] = 0
                    cnt2[cov2] += 1
                
                if len(sel_idx2) < best_len:
                    current_indices = sel_idx2[:]
                    current_counts = cnt2.copy()
                    current_mask = sel_mask2.copy()
                    T = 5.0
                    stagnation = 0
                else:
                    T = max(T * cooling, min_T)

    final_indices = best_solution
    final_counts = best_counts.copy()
    final_mask = best_mask.copy()

    for b_idx in final_indices[:]:
        if not final_mask[b_idx]:
            continue
        covered_indices = all_covered[b_idx]
        if np.all(final_counts[covered_indices] > 1):
            final_mask[b_idx] = False
            final_indices.remove(b_idx)
            final_counts[covered_indices] -= 1

    verify_counts = np.zeros(num_tsubs, dtype=np.int32)
    for b_idx in final_indices:
        covered_indices = all_covered[b_idx]
        verify_counts[covered_indices] += 1

    if not np.all(verify_counts >= 1):
        # Fallback to a valid solution
        fallback = greedy_construction(0)
        if fallback:
            final_indices = fallback
        else:
            final_indices = selected_indices if 'selected_indices' in locals() else []

    return get_blocks(final_indices)

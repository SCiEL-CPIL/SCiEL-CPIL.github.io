import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for conversion and cleanup
    
    total_space = q ** n
    
    # Precompute powers of q
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
    
    # Precompute digits for all indices
    digits_all = []
    for j in range(total_space):
        d = [0] * n
        val = j
        for pos in range(n):
            d[pos] = val % q
            val //= q
        digits_all.append(d)
        
    # Precompute neighbor indices and masks
    neighbor_indices = []
    neighbor_masks = [0] * total_space
    
    for j in range(total_space):
        indices = [j]
        mask = 1 << j
        d = digits_all[j]
        if radius >= 1:
            for pos in range(n):
                old_val = d[pos]
                p = powers[pos]
                for new_val in range(q):
                    if new_val == old_val:
                        continue
                    new_idx = j + (new_val - old_val) * p
                    indices.append(new_idx)
                    mask |= (1 << new_idx)
        neighbor_indices.append(indices)
        neighbor_masks[j] = mask

    all_words = list(range(total_space))
    full_mask = (1 << total_space) - 1
    
    def greedy_init(seed=None):
        """Fast greedy set cover using bitsets."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        remaining_mask = full_mask
        selected = []
        
        for _ in range(total_space):
            if remaining_mask == 0:
                break
            if time.time() > deadline - 10.0:
                break
                
            best_gain = -1
            best_candidates = []
            
            for w in range(total_space):
                gain = (neighbor_masks[w] & remaining_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [w]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(w)
                    
            if best_gain <= 0:
                break
                
            w = rng.choice(best_candidates)
            selected.append(w)
            remaining_mask &= ~neighbor_masks[w]
            
        return selected

    def local_search(initial_selected, time_budget, seed):
        """Run simulated annealing local search from an initial solution."""
        rng = random.Random(seed)
        
        selected_set = set(initial_selected)
        coverage_counts = [0] * total_space
        for w in selected_set:
            for idx in neighbor_indices[w]:
                coverage_counts[idx] += 1
                
        uncovered_count = sum(1 for c in coverage_counts if c == 0)
        current_cost = len(selected_set) + 1000 * uncovered_count
        
        best_cost = current_cost
        best_set = set(selected_set)
        
        T = 10.0
        T_min = 0.01
        alpha = 0.9995
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            r = rng.random()
            
            if r < 0.3:
                # Add
                w = rng.choice(all_words)
                if w in selected_set:
                    continue
                gain = 0
                for idx in neighbor_indices[w]:
                    if coverage_counts[idx] == 0:
                        gain += 1
                delta_cost = 1 - 1000 * gain
                if delta_cost < 0 or rng.random() < pow(2.718, -delta_cost / T):
                    selected_set.add(w)
                    for idx in neighbor_indices[w]:
                        if coverage_counts[idx] == 0:
                            uncovered_count -= 1
                        coverage_counts[idx] += 1
                    current_cost += delta_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_set = set(selected_set)
                        
            elif r < 0.6:
                # Remove
                if not selected_set:
                    continue
                w = rng.choice(list(selected_set))
                loss = 0
                for idx in neighbor_indices[w]:
                    if coverage_counts[idx] == 1:
                        loss += 1
                delta_cost = -1 + 1000 * loss
                if delta_cost < 0 or rng.random() < pow(2.718, -delta_cost / T):
                    selected_set.remove(w)
                    for idx in neighbor_indices[w]:
                        if coverage_counts[idx] == 1:
                            uncovered_count += 1
                        coverage_counts[idx] -= 1
                    current_cost += delta_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_set = set(selected_set)
                        
            else:
                # Swap
                if len(selected_set) < 2:
                    continue
                w1 = rng.choice(list(selected_set))
                w2 = rng.choice(all_words)
                if w1 == w2:
                    continue
                
                newly_uncovered = 0
                for i in neighbor_indices[w1]:
                    if coverage_counts[i] == 1:
                        if not (neighbor_masks[w2] >> i) & 1:
                            newly_uncovered += 1
                            
                newly_covered = 0
                for i in neighbor_indices[w2]:
                    if coverage_counts[i] == 0:
                        if not (neighbor_masks[w1] >> i) & 1:
                            newly_covered += 1
                            
                delta_uncovered = newly_uncovered - newly_covered
                delta_cost = 1000 * delta_uncovered
                
                if delta_cost < 0 or rng.random() < pow(2.718, -delta_cost / T):
                    selected_set.remove(w1)
                    selected_set.add(w2)
                    uncovered_count += delta_uncovered
                    for i in neighbor_indices[w1]:
                        coverage_counts[i] -= 1
                    for i in neighbor_indices[w2]:
                        coverage_counts[i] += 1
                    current_cost += delta_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_set = set(selected_set)

            T *= alpha
            if T < T_min:
                T = T_min

        return best_set, best_cost

    def repair_cover(final_set):
        """Ensure the set is a valid cover by greedily adding missing words."""
        test_counts = [0] * total_space
        for w in final_set:
            for idx in neighbor_indices[w]:
                test_counts[idx] += 1
                
        uncovered_indices = [i for i in range(total_space) if test_counts[i] == 0]
        
        if uncovered_indices:
            remaining = set(uncovered_indices)
            while remaining:
                best_gain = -1
                best_w = -1
                for w in range(total_space):
                    if w in final_set:
                        continue
                    gain = 0
                    for idx in neighbor_indices[w]:
                        if idx in remaining:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_w = w
                if best_gain <= 0:
                    break
                final_set.add(best_w)
                for idx in neighbor_indices[best_w]:
                    if idx in remaining:
                        remaining.discard(idx)
                        
        return final_set

    # Multi-start strategy:
    # Divide the time budget among several independent local search runs.
    # Each run starts from a greedy solution with a different random seed.
    
    num_starts = 4
    time_per_start = (deadline - time.time() - 5.0) / num_starts
    
    best_final_set = None
    best_final_cost = float('inf')
    
    for i in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        seed = 42 + i * 1000
        init_sol = greedy_init(seed=seed)
        if not init_sol:
            init_sol = [0]
            
        # Run local search for the allocated time
        ls_set, ls_cost = local_search(init_sol, time_per_start, seed)
        
        # Repair to ensure validity
        repaired_set = repair_cover(set(ls_set))
        
        # Calculate final cost (should be valid, so uncovered=0)
        final_count = len(repaired_set)
        if final_count < best_final_cost:
            best_final_cost = final_count
            best_final_set = repaired_set
            
    if best_final_set is None:
        # Fallback: just use a greedy solution
        best_final_set = set(greedy_init(seed=42))
        best_final_set = repair_cover(best_final_set)
        
    # Convert indices to lists of digits
    result = []
    for idx in best_final_set:
        result.append([int(d) for d in digits_all[idx]])
        
    return result

if __name__ == "__main__":
    # Test
    res = construct(3, 7, 1)
    print(len(res))

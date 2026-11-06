import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin before 90s kill
    
    n = 512
    k = 4
    
    # Precompute all 4-APs in [1, n]
    # A 4-AP is (a, a+d, a+2d, a+3d)
    all_aps = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            all_aps.append((a, a+d, a+2*d, a+3*d))
    
    all_aps_np = np.array(all_aps, dtype=np.int32)
    num_aps = len(all_aps_np)
    
    def is_valid_bitset(bitset):
        """
        bitset: numpy array of shape (n+1,) with 1s at selected indices.
        Returns True if no 4-AP is fully contained.
        """
        # Extract bits for all AP terms
        # terms shape: (num_aps, 4)
        terms = bitset[all_aps_np]
        # Check if any row has all 1s
        # sum along axis 1, check if any sum is 4
        return not np.any(np.sum(terms, axis=1) == 4)
    
    def get_size(bitset):
        return np.sum(bitset)
    
    def greedy_seed(mode="standard"):
        """
        Generate a valid 4-AP-free set using a greedy approach.
        mode: 'standard' (1..n), 'reverse' (n..1), 'random' (shuffled)
        """
        selected = np.zeros(n + 1, dtype=np.int8)
        candidates = list(range(1, n + 1))
        
        if mode == "reverse":
            candidates.reverse()
        elif mode == "random":
            random.shuffle(candidates)
        elif mode == "density":
            # Prioritize numbers with fewer conflicts? 
            # Simple heuristic: numbers with prime factors or specific modular properties?
            # For now, just a weighted shuffle based on index or random
            random.shuffle(candidates)
            
        for x in candidates:
            # Check if adding x creates a 4-AP
            # We can use the precomputed APs to check conflicts quickly?
            # Actually, for greedy, we just need to check if x completes any AP with existing elements.
            # Since selected is small initially, set membership is fast.
            # But for speed, let's use a simple set.
            sel_set = set(np.nonzero(selected)[0])
            
            valid = True
            # Check if x is part of any 4-AP with 3 other elements in sel_set
            # This is O(n^2) if done naively for each x.
            # Optimization: Precompute for each x, the list of APs it belongs to.
            # But we don't have that precomputed.
            # Given n=512, O(n^2) per addition is ~260k ops. 512 additions -> 130M ops. Too slow in Python.
            
            # Better: Use the bitset check? No, bitset check is global.
            # Let's stick to the fast set-based check but optimize the loop.
            
            # x as 1st: x, x+d, x+2d, x+3d
            for d in range(1, (n - x) // 3 + 1):
                if (x + d) in sel_set and (x + 2*d) in sel_set and (x + 3*d) in sel_set:
                    valid = False
                    break
            if not valid: continue
            
            # x as 2nd: x-d, x, x+d, x+2d
            for d in range(1, (x - 1) // 3 + 1):
                a = x - d
                if a in sel_set and (x + d) in sel_set and (x + 2*d) in sel_set:
                    valid = False
                    break
            if not valid: continue
            
            # x as 3rd: x-2d, x-d, x, x+d
            for d in range(1, (x - 1) // 2 + 1):
                a = x - 2*d
                if a in sel_set and (a + d) in sel_set and (x + d) in sel_set:
                    valid = False
                    break
            if not valid: continue
            
            # x as 4th: x-3d, x-2d, x-d, x
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3*d
                if a in sel_set and (a + d) in sel_set and (a + 2*d) in sel_set:
                    valid = False
                    break
            
            if valid:
                selected[x] = 1
                sel_set.add(x)
                
        return selected

    def simulated_annealing(initial_bitset, max_time):
        """
        Improve the set using Simulated Annealing with 2-for-1 swaps.
        """
        current_bitset = initial_bitset.copy()
        current_size = get_size(current_bitset)
        best_bitset = current_bitset.copy()
        best_size = current_size
        
        # Temperature
        T = 1.0
        cooling_rate = 0.99
        
        unselected = set(np.nonzero(current_bitset == 0)[0])
        selected = set(np.nonzero(current_bitset == 1)[0])
        
        while time.time() < max_time and T > 0.01:
            # Try to find a move
            # Move 1: Add x (if possible)
            # Move 2: Swap (remove y, add x) - maintains size
            # Move 3: 2-for-1 (remove y, add x1, x2) - increases size by 1
            
            # We want to maximize size.
            # If we can add, we do it.
            # If not, we swap to rearrange.
            
            # Randomly pick a move type
            if unselected:
                x = random.choice(list(unselected))
                
                # Try Add
                test_bitset = current_bitset.copy()
                test_bitset[x] = 1
                if is_valid_bitset(test_bitset):
                    current_bitset = test_bitset
                    current_size += 1
                    unselected.remove(x)
                    selected.add(x)
                    if current_size > best_size:
                        best_bitset = current_bitset.copy()
                        best_size = current_size
                    continue
                
                # Try 2-for-1: remove y, add x and x2
                if len(unselected) >= 2:
                    # Pick another x2
                    candidates = list(unselected)
                    candidates.remove(x)
                    if candidates:
                        x2 = random.choice(candidates)
                        # Pick y from selected
                        if selected:
                            y = random.choice(list(selected))
                            
                            test_bitset = current_bitset.copy()
                            test_bitset[y] = 0
                            test_bitset[x] = 1
                            test_bitset[x2] = 1
                            
                            if is_valid_bitset(test_bitset):
                                current_bitset = test_bitset
                                current_size += 1
                                unselected.remove(x)
                                unselected.remove(x2)
                                unselected.add(y)
                                selected.remove(y)
                                selected.add(x)
                                selected.add(x2)
                                
                                if current_size > best_size:
                                    best_bitset = current_bitset.copy()
                                    best_size = current_size
                                continue
                            
                            # If invalid, try a swap (remove y, add x) to keep size same but change structure
                            # This helps escape local optima
                            test_bitset = current_bitset.copy()
                            test_bitset[y] = 0
                            test_bitset[x] = 1
                            
                            if is_valid_bitset(test_bitset):
                                # Accept swap with probability based on T? 
                                # Since size is same, we accept to explore.
                                # But we need to be careful not to oscillate.
                                # For now, accept with prob 0.5 or always? 
                                # Let's accept to explore.
                                current_bitset = test_bitset
                                # Size unchanged
                                unselected.remove(x)
                                unselected.add(y)
                                selected.remove(y)
                                selected.add(x)
                                continue
                            
            T *= cooling_rate
            
        return best_bitset, best_size

    # Multi-start strategy
    random.seed(42)
    
    best_overall_bitset = None
    best_overall_size = 0
    
    # Generate diverse seeds
    seeds = [
        greedy_seed("standard"),
        greedy_seed("reverse"),
        greedy_seed("random"),
    ]
    
    # Allocate time to each seed's SA
    time_per_seed = (deadline - (time.time() - start_time)) / len(seeds) - 2
    
    for seed_bitset in seeds:
        if time.time() > deadline - 5:
            break
            
        seed_size = get_size(seed_bitset)
        if seed_size > best_overall_size:
            best_overall_bitset = seed_bitset.copy()
            best_overall_size = seed_size
            
        # Run SA for a portion of the time
        # Ensure we don't exceed deadline
        max_sa_time = min(time_per_seed, deadline - (time.time() - start_time) - 2)
        if max_sa_time > 1:
            sa_bitset, sa_size = simulated_annealing(seed_bitset, max_sa_time)
            if sa_size > best_overall_size:
                best_overall_bitset = sa_bitset
                best_overall_size = sa_size
        
        # Update best seed for next iteration? No, independent starts are better for diversity.
        
    # Final polish: Run SA on the best found so far with remaining time
    if best_overall_bitset is not None:
        remaining_time = deadline - (time.time() - start_time) - 2
        if remaining_time > 1:
            final_bitset, final_size = simulated_annealing(best_overall_bitset, remaining_time)
            if final_size > best_overall_size:
                best_overall_bitset = final_bitset
                best_overall_size = final_size
    
    if best_overall_bitset is None:
        # Fallback
        return [1]
        
    return sorted(np.nonzero(best_overall_bitset)[0].tolist())

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    s = set(res)
    valid = True
    for a in s:
        for d in range(1, (512 - a) // 3 + 1):
            if (a+d) in s and (a+2*d) in s and (a+3*d) in s:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")

import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 90s limit, evaluator kills at 90s, but 60s is mentioned in prompt. Let's be safe with 58s.
    
    # Helper: bit operations
    def val_to_bit(x):
        return 1 << (x - 1)
    
    # Precompute 4-AP conflict masks
    # forbidden_masks[v] is a list of masks. If (S_mask & mask) == mask, then v cannot be added.
    forbidden_masks = [[] for _ in range(n + 1)]
    
    for v in range(1, n + 1):
        masks = []
        # Case 1: v is 1st term: v, v+d, v+2d, v+3d
        # Need v+d, v+2d, v+3d in S
        for d in range(1, n // 3 + 1):
            if v + 3 * d <= n:
                m = val_to_bit(v + d) | val_to_bit(v + 2 * d) | val_to_bit(v + 3 * d)
                masks.append(m)
        # Case 2: v is 2nd term: v-d, v, v+d, v+2d
        # Need v-d, v+d, v+2d in S
        for d in range(1, n // 3 + 1):
            if v - d >= 1 and v + 2 * d <= n:
                m = val_to_bit(v - d) | val_to_bit(v + d) | val_to_bit(v + 2 * d)
                masks.append(m)
        # Case 3: v is 3rd term: v-2d, v-d, v, v+d
        # Need v-2d, v-d, v+d in S
        for d in range(1, n // 3 + 1):
            if v - 2 * d >= 1 and v + d <= n:
                m = val_to_bit(v - 2 * d) | val_to_bit(v - d) | val_to_bit(v + d)
                masks.append(m)
        # Case 4: v is 4th term: v-3d, v-2d, v-d, v
        # Need v-3d, v-2d, v-d in S
        for d in range(1, n // 3 + 1):
            if v - 3 * d >= 1:
                m = val_to_bit(v - 3 * d) | val_to_bit(v - 2 * d) | val_to_bit(v - d)
                masks.append(m)
        forbidden_masks[v] = masks
        
    # Precompute 3-AP masks for delta calculation
    # ap3_masks[u] is a list of masks. Each mask represents the other 2 elements of a 3-AP containing u.
    # If (S_mask & mask) == mask, that 3-AP is present.
    ap3_masks = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        masks = []
        # u is 1st: u, u+d, u+2d. Other: u+d, u+2d.
        for d in range(1, (n - u) // 2 + 1):
            masks.append(val_to_bit(u + d) | val_to_bit(u + 2 * d))
        # u is 2nd: u-d, u, u+d. Other: u-d, u+d.
        for d in range(1, min(u - 1, n - u) + 1):
            masks.append(val_to_bit(u - d) | val_to_bit(u + d))
        # u is 3rd: u-2d, u-d, u. Other: u-2d, u-d.
        for d in range(1, (u - 1) // 2 + 1):
            masks.append(val_to_bit(u - 2 * d) | val_to_bit(u - d))
        ap3_masks[u] = masks

    def count_3aps(S_mask):
        """Count 3-APs in S_mask. Used for initialization and occasional verification."""
        # Efficient counting using bit operations is complex for arbitrary sets.
        # For n=256, a simple loop over a, d is O(N^2) ~ 65k ops.
        # But we can do better.
        # Let's just use the mask sum method if we precomputed something?
        # Actually, for the GA, we need fast evaluation.
        # We can maintain the count incrementally during mutations.
        # For initialization, we can compute it once.
        
        # Alternative: Precompute all 3-APs as masks.
        # There are ~ N^2/2 ~ 32k 3-APs.
        # Counting how many are subsets of S is O(num_aps) per check.
        # 32k checks is fast in Python if optimized? Maybe.
        # But let's stick to incremental updates for the GA steps.
        # For the initial population, we can just compute it directly.
        
        count = 0
        # Iterate a, d
        # a in 1..n, d in 1..(n-a)//2
        # Check if a, a+d, a+2d in S
        # This is ~ N^2/2 iterations.
        # 256^2 / 2 = 32768 iterations.
        # Each iteration is 3 bit checks. ~100k ops.
        # Acceptable for ~50 initial individuals.
        
        for a in range(1, n + 1):
            if not ((S_mask >> (a-1)) & 1):
                continue
            max_d = (n - a) // 2
            for d in range(1, max_d + 1):
                if ((S_mask >> (a+d-1)) & 1) and ((S_mask >> (a+2*d-1)) & 1):
                    count += 1
        return count

    def can_add(S_mask, v):
        """Check if v can be added to S_mask without creating a 4-AP."""
        for m in forbidden_masks[v]:
            if (S_mask & m) == m:
                return False
        return True

    def calc_delta(S_mask, u, v):
        """
        Calculate change in 3-AP count when removing u and adding v to S_mask.
        S_mask contains u. v is not in S_mask.
        Returns gained - lost.
        """
        # Lost: 3-APs containing u in S
        lost = 0
        for m in ap3_masks[u]:
            if (S_mask & m) == m:
                lost += 1
                
        # Gained: 3-APs containing v in S \ {u}
        S_without_u = S_mask & ~val_to_bit(u)
        gained = 0
        for m in ap3_masks[v]:
            if (S_without_u & m) == m:
                gained += 1
                
        return gained - lost

    # --- Genetic Algorithm Setup ---
    
    # 1. Generate Initial Population
    # Strategy: Greedy construction with random ordering
    def greedy_random():
        S_mask = 0
        order = list(range(1, n + 1))
        random.shuffle(order)
        for x in order:
            if can_add(S_mask, x):
                S_mask |= val_to_bit(x)
        return S_mask

    pop_size = 20
    population = []
    for _ in range(pop_size):
        S_mask = greedy_random()
        count = count_3aps(S_mask)
        population.append((S_mask, count))
        
    # Sort population by fitness (count)
    population.sort(key=lambda x: x[1], reverse=True)
    
    best_mask = population[0][0]
    best_count = population[0][1]
    best_set = {i+1 for i in range(n) if (best_mask >> i) & 1}

    # 2. Evolutionary Loop
    # Parameters
    elite_size = 5
    mutation_rate = 0.3
    crossover_rate = 0.7
    
    # We will run generations until time limit.
    # Each generation creates pop_size offspring.
    
    while time.time() - start_time < time_limit:
        # Select parents
        # Tournament selection or roulette wheel?
        # Let's use tournament selection of size 3.
        parents = []
        for _ in range(pop_size):
            # Pick 3 random individuals, take the best
            candidates = random.sample(population, 3)
            parents.append(max(candidates, key=lambda x: x[1]))
            
        new_population = []
        
        # Elitism: Keep top elite_size
        for i in range(elite_size):
            new_population.append(population[i])
            
        # Generate offspring
        while len(new_population) < pop_size:
            p1 = parents[len(new_population) - elite_size][0]
            p2 = parents[(len(new_population) - elite_size + 1) % pop_size][0]
            
            if random.random() < crossover_rate:
                # Crossover: Uniform crossover on bitmasks
                # For each bit, pick from p1 or p2
                child_mask = 0
                for i in range(n):
                    bit_p1 = (p1 >> i) & 1
                    bit_p2 = (p2 >> i) & 1
                    if bit_p1:
                        child_mask |= val_to_bit(i+1)
                    elif bit_p2:
                        # 50% chance to take from p2
                        if random.random() < 0.5:
                            child_mask |= val_to_bit(i+1)
            else:
                child_mask = p1 | p2 # Union is a valid 4-AP free set?
                # Wait! Union of two 4-AP free sets is NOT necessarily 4-AP free.
                # {1,2,3} is 4-AP free. {4,5,6} is 4-AP free.
                # Union {1,2,3,4,5,6} contains 1,2,3,4? No, 1,2,3,4 is not 4-AP (1,2,3,4 diff 1, yes it is).
                # So union is INVALID.
                # Intersection IS valid (subset of valid set).
                # So we should use Intersection for crossover to maintain validity.
                # Or we need a repair step.
                
                # Let's use Intersection for safety, then mutate to grow.
                child_mask = p1 & p2
            
            # Mutation
            # Randomly flip bits?
            # Flipping 0->1 requires validity check.
            # Flipping 1->0 is always valid.
            
            # Strategy: 
            # 1. Try to add random elements (0->1) if valid.
            # 2. Try to remove random elements (1->1) to allow other additions?
            #    Or just do simple random restarts of parts of the set?
            
            # Let's do:
            # With prob 0.5, try to add 1-3 random elements.
            # With prob 0.5, try to remove 1-3 random elements.
            
            if random.random() < 0.5:
                # Add
                num_adds = random.randint(1, 3)
                candidates = [x for x in range(1, n+1) if not ((child_mask >> (x-1)) & 1)]
                random.shuffle(candidates)
                for x in candidates[:num_adds]:
                    if can_add(child_mask, x):
                        child_mask |= val_to_bit(x)
            else:
                # Remove
                num_rems = random.randint(1, 3)
                in_set = [x for x in range(1, n+1) if (child_mask >> (x-1)) & 1]
                random.shuffle(in_set)
                for x in in_set[:num_rems]:
                    child_mask &= ~val_to_bit(x)
            
            # Evaluate offspring
            # We need to compute the count.
            # Since we mutated, we don't have a direct delta from a known parent count easily if we did multiple flips.
            # We can compute from scratch (fast enough for 20 individuals * few generations?)
            # Or we can try to track delta for single flips.
            # Given the complexity of tracking multi-flip deltas, and n=256, 
            # count_3aps is ~100k ops. 
            # If we do 20 offspring per generation, that's 2M ops.
            # In Python, 2M simple ops might take 0.1-0.2s.
            # So we can do ~50-100 generations in 10s?
            # Total time 58s. So ~300-500 generations.
            # This seems reasonable.
            
            child_count = count_3aps(child_mask)
            new_population.append((child_mask, child_count))
            
            # Update best
            if child_count > best_count:
                best_count = child_count
                best_mask = child_mask
                best_set = {i+1 for i in range(n) if (best_mask >> i) & 1}
        
        population = new_population
        # Sort to maintain elite order for next iteration?
        # Not strictly necessary if we use tournament selection on the unsorted list,
        # but good for debugging.
        population.sort(key=lambda x: x[1], reverse=True)

    return sorted(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])

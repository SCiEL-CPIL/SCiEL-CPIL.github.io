import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    # Evaluator kills at 90s, task says 60s. 
    # Use 58s to be safe.
    time_limit = 58.0
    
    # --- Precomputation ---
    
    # 1. Precompute 4-AP conflicts.
    # For each element v, store the list of "witness" masks.
    # If (S & mask) == mask, then adding v creates a 4-AP.
    # mask is a 3-bit integer representing the other 3 elements.
    forbidden_masks = [[] for _ in range(n + 1)]
    for v in range(1, n + 1):
        masks = []
        # v is 1st: v, v+d, v+2d, v+3d
        for d in range(1, (n - v) // 3 + 1):
            u1, u2, u3 = v + d, v + 2 * d, v + 3 * d
            if u3 <= n:
                m = (1 << (u1 - 1)) | (1 << (u2 - 1)) | (1 << (u3 - 1))
                masks.append(m)
        # v is 2nd: v-d, v, v+d, v+2d
        for d in range(1, min(v - 1, (n - v) // 2) + 1):
            u1, u2, u3 = v - d, v + d, v + 2 * d
            if u1 >= 1 and u3 <= n:
                m = (1 << (u1 - 1)) | (1 << (u2 - 1)) | (1 << (u3 - 1))
                masks.append(m)
        # v is 3rd: v-2d, v-d, v, v+d
        for d in range(1, (v - 1) // 2 + 1):
            u1, u2, u3 = v - 2 * d, v - d, v + d
            if u1 >= 1 and u3 <= n:
                m = (1 << (u1 - 1)) | (1 << (u2 - 1)) | (1 << (u3 - 1))
                masks.append(m)
        # v is 4th: v-3d, v-2d, v-d, v
        for d in range(1, (v - 1) // 3 + 1):
            u1, u2, u3 = v - 3 * d, v - 2 * d, v - d
            if u1 >= 1:
                m = (1 << (u1 - 1)) | (1 << (u2 - 1)) | (1 << (u3 - 1))
                masks.append(m)
        forbidden_masks[v] = masks

    # 2. Precompute 3-AP contributions.
    # For each element v, store the list of "witness" masks.
    # If (S & mask) == mask, then adding v completes a 3-AP.
    # mask is a 2-bit integer representing the other 2 elements.
    ap3_masks = [[] for _ in range(n + 1)]
    for v in range(1, n + 1):
        masks = []
        # v is 1st: v, v+d, v+2d
        for d in range(1, (n - v) // 2 + 1):
            u1, u2 = v + d, v + 2 * d
            if u2 <= n:
                m = (1 << (u1 - 1)) | (1 << (u2 - 1))
                masks.append(m)
        # v is 2nd: v-d, v, v+d
        for d in range(1, min(v - 1, n - v) + 1):
            u1, u2 = v - d, v + d
            if u1 >= 1 and u2 <= n:
                m = (1 << (u1 - 1)) | (1 << (u2 - 1))
                masks.append(m)
        # v is 3rd: v-2d, v-d, v
        for d in range(1, (v - 1) // 2 + 1):
            u1, u2 = v - 2 * d, v - d
            if u1 >= 1:
                m = (1 << (u1 - 1)) | (1 << (u2 - 1))
                masks.append(m)
        ap3_masks[v] = masks

    # Helper: Check if v can be added to S (represented as bitmask)
    def can_add(S_mask, v):
        for m in forbidden_masks[v]:
            if (S_mask & m) == m:
                return False
        return True

    # Helper: Calculate gain in 3-APs if v is added to S
    def calc_gain(S_mask, v):
        gain = 0
        for m in ap3_masks[v]:
            if (S_mask & m) == m:
                gain += 1
        return gain

    best_set = set()
    best_count = 0
    best_mask = 0

    # --- Strategy: Backtracking with Heuristic Ordering ---
    # We perform a depth-first search (backtracking) to build the set.
    # At each step, we choose the next element to add.
    # Heuristic: Prioritize elements that have high "potential" (many 3-APs with current S) 
    # but low "conflict" (few forbidden masks that are close to being satisfied).
    # However, simple backtracking on 256 elements is too slow if we explore too deeply.
    # Instead, we use a "Greedy Backtracking" approach:
    # 1. Order elements by a heuristic score.
    # 2. Iterate through the order.
    # 3. If element can be added, add it.
    # 4. If adding it leads to a dead end (no more elements can be added), backtrack.
    # To keep it fast, we limit the depth of backtracking and use randomization in the order.
    
    # A better approach for this specific problem (maximizing 3-APs subject to no 4-APs)
    # is often found via "Max-SAT" style local search or specialized constructions.
    # Given the time limit, a robust Multi-Start Greedy with Limited Backtracking is safer.
    
    # Let's try a different structural change: 
    # **Iterative Improvement with Conflict Resolution**
    # 1. Start with a dense random set.
    # 2. Identify all 4-APs.
    # 3. Remove elements from these 4-APs to break them.
    # 4. Try to add elements back in to maximize 3-APs.
    
    # Actually, the most effective structural change for "Maximize 3-APs, No 4-APs"
    # is often to start from a known good structure (like a Cap Set or similar) 
    # or use a **Genetic Algorithm** with a strong fitness function.
    
    # Let's implement a **Genetic Algorithm** approach.
    # Representation: Bitmask (int).
    # Fitness: Count of 3-APs - (Large Penalty if any 4-AP exists).
    # Since we need strictly no 4-APs, we can use a penalty method.
    # Penalty = 10000 * (number of 4-APs).
    # Fitness = count_3aps - penalty.
    
    def count_4aps(S_mask):
        count = 0
        # A 4-AP is {a, a+d, a+2d, a+3d}.
        # To count distinct 4-APs, we can iterate a, d.
        # But checking all a, d is O(n^2).
        # Faster: For each v in S, check if it's part of a 4-AP? 
        # No, we just need to know if ANY exist for the penalty.
        # Or count them for gradient.
        # Let's just check validity. If invalid, penalty is huge.
        # But for GA, we need a smooth fitness.
        # Let's count the number of 4-APs.
        # Iterate over all possible 4-APs in [1, 256].
        # There are ~ n^2/6 such APs. 256^2/6 ~ 10k.
        # Checking each is O(1) with bitmasks.
        # Precompute all 4-AP masks.
        pass

    # Precompute all 4-AP masks for penalty calculation
    four_ap_masks = []
    for a in range(1, n - 3):
        for d in range(1, (n - a) // 3 + 1):
            a3 = a + 3 * d
            if a3 <= n:
                m = (1 << (a - 1)) | (1 << (a + d - 1)) | (1 << (a + 2 * d - 1)) | (1 << (a + 3 * d - 1))
                four_ap_masks.append(m)
                
    def get_fitness(S_mask):
        # Count 3-APs
        c3 = 0
        for v in range(1, n + 1):
            if (S_mask >> (v - 1)) & 1:
                for m in ap3_masks[v]:
                    if (S_mask & m) == m:
                        c3 += 1
        # Count 4-APs
        c4 = 0
        for m in four_ap_masks:
            if (S_mask & m) == m:
                c4 += 1
        # Fitness: Maximize 3-APs, minimize 4-APs.
        # If c4 > 0, fitness is very low.
        return c3 - 10000 * c4

    # GA Implementation
    population_size = 100
    population = []
    
    # Initialize population with random valid sets (or greedy sets)
    for _ in range(population_size):
        S_mask = 0
        S_set = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for v in candidates:
            if can_add(S_mask, v):
                S_mask |= (1 << (v - 1))
                S_set.add(v)
        population.append(S_mask)
        
    best_in_pop = max(population, key=get_fitness)
    best_count = get_fitness(best_in_pop)
    best_mask = best_in_pop
    best_set = set(i + 1 for i in range(n) if (best_mask >> i) & 1)
    
    while time.time() - start_time < time_limit - 2.0:
        # Selection: Tournament
        new_population = []
        for _ in range(population_size // 2):
            p1 = random.choice(population)
            p2 = random.choice(population)
            parent = p1 if get_fitness(p1) > get_fitness(p2) else p2
            new_population.append(parent)
            
        # Crossover and Mutation
        for i in range(0, len(new_population), 2):
            if i + 1 >= len(new_population):
                break
            p1 = new_population[i]
            p2 = new_population[i + 1]
            
            # Uniform Crossover
            child_mask = 0
            for bit in range(n):
                if random.random() < 0.5:
                    child_mask |= (p1 >> bit) & 1
                else:
                    child_mask |= (p2 >> bit) & 1
                    child_mask <<= 1
                    child_mask |= (p2 >> bit) & 1
                    # Wait, this loop is wrong for building a mask.
                    # Let's do it properly.
            
            # Proper Uniform Crossover
            child_mask = 0
            for bit in range(n):
                if random.random() < 0.5:
                    if (p1 >> bit) & 1:
                        child_mask |= (1 << bit)
                else:
                    if (p2 >> bit) & 1:
                        child_mask |= (1 << bit)
                        
            # Mutation: Flip a few bits
            for _ in range(5):
                bit = random.randint(0, n - 1)
                child_mask ^= (1 << bit)
                
            new_population.append(child_mask)
            
        # Evaluate new population
        for i, mask in enumerate(new_population):
            if get_fitness(mask) > get_fitness(best_mask):
                best_mask = mask
                best_count = get_fitness(mask)
                best_set = set(i + 1 for i in range(n) if (best_mask >> i) & 1)
                
        # Replace population with new one (elitism: keep best)
        population = new_population
        population[0] = best_mask
        
    # Final verification
    if not best_set:
        return [1]
        
    return sorted(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])

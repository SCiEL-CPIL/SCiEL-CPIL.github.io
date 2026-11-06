import time
from itertools import combinations

# Precompute all 4-subsets of [1, 16]
N = 16
K = 4
ALL_COMBOS = list(combinations(range(1, N + 1), K))
NUM_SETS = len(ALL_COMBOS)

# Convert to bitmasks for fast intersection
ALL_SETS = [0] * NUM_SETS
for i, c in enumerate(ALL_COMBOS):
    mask = 0
    for x in c:
        mask |= (1 << (x - 1))
    ALL_SETS[i] = mask

# Precompute pairwise intersections as a matrix of masks
# inter[i][j] = ALL_SETS[i] & ALL_SETS[j]
# This is a 1820 x 1820 matrix of integers.
# Memory: 1820 * 1820 * 8 bytes ~ 26 MB. Acceptable.
# We can store it as a list of lists or a flat array.
# Let's use a list of lists for simplicity, or numpy if available?
# Standard library is preferred for portability, but numpy is allowed.
# Let's stick to standard library lists of lists or a flat list.
# A flat list of size 1820*1820 is ~3.3M integers.
# Actually, we only need to check if adding set k creates a sunflower with i, j.
# Condition: (A&B == A&C == B&C)
# Let C be the new set.
# We need to check for all pairs (i, j) in the current family:
#   (S_i & S_j) == (S_i & S_k) == (S_j & S_k)
#
# Let's precompute inter[i][j] for all i, j.
inter = [[0] * NUM_SETS for _ in range(NUM_SETS)]
for i in range(NUM_SETS):
    mi = ALL_SETS[i]
    for j in range(i + 1, NUM_SETS):
        mj = ALL_SETS[j]
        ij = mi & mj
        inter[i][j] = ij
        inter[j][i] = ij

def can_add(idx, family_indices):
    """
    Check if adding set at index idx to family_indices creates a sunflower.
    family_indices is a list of indices into ALL_SETS.
    """
    new_mask = ALL_SETS[idx]
    m = len(family_indices)
    
    # Optimization: If m is small, just check all pairs.
    # If m is large, we can be smarter.
    # For now, O(m^2) check.
    for i in range(m):
        a_idx = family_indices[i]
        # Precompute intersection of new set with a
        inter_new_a = new_mask & ALL_SETS[a_idx]
        
        for j in range(i + 1, m):
            b_idx = family_indices[j]
            # Check if (a & b) == (a & new) == (b & new)
            # We have inter[a_idx][b_idx]
            # We need (b & new) to be equal to inter_new_a
            # And inter[a_idx][b_idx] to be equal to inter_new_a
            
            if inter[a_idx][b_idx] == inter_new_a:
                # Now check if (b & new) == inter_new_a
                if (ALL_SETS[b_idx] & new_mask) == inter_new_a:
                    return False
    return True

def greedy_randomized(seed=0):
    import random
    random.seed(seed)
    indices = list(range(NUM_SETS))
    random.shuffle(indices)
    
    family = []
    for idx in indices:
        if can_add(idx, family):
            family.append(idx)
    return family

def greedy_min_degree(seed=0):
    """
    Greedy algorithm that selects the set with the minimum number of conflicts
    or simply the first available set in a shuffled order, but we can try
    a different heuristic.
    
    Actually, the "minimum degree" heuristic in the context of independent sets
    in a graph (or hypergraph) usually means picking the vertex with the fewest
    neighbors. Here, "neighbors" are sets that form a sunflower with any two
    existing sets? No, that's dynamic.
    
    A common heuristic for independent sets is:
    1. Start with empty set.
    2. At each step, pick the vertex with the fewest connections to other *available* vertices.
    But here the constraint is 3-uniform.
    
    Let's try a different approach: 
    Instead of random order, we can try to order sets by some property.
    For example, sets with fewer "potential sunflowers" with other sets.
    But that's hard to precompute for 3-uniform.
    
    Let's stick to randomized greedy but with multiple restarts and maybe
    a local search improvement step.
    """
    pass

def local_search(family, time_limit=5.0):
    """
    Try to improve the family by swapping sets.
    1. Remove a set A from the family.
    2. Try to add a set B that was previously rejected or not in the family.
    3. If B can be added, keep the new family.
    4. If size increases, great. If size is same, maybe it helps future additions?
       Actually, we want to maximize size.
       
    A better local search for independent set:
    - Try to add a set that is currently blocked.
    - If it's blocked by pairs (i, j), remove one of i or j, then add the new set.
    - This keeps size same or increases if we can add more later?
    - Actually, removing one and adding one keeps size same.
    - To increase size, we need to find a set that can be added *after* removing some sets,
      such that we can then add *more* sets.
    
    This is complex. Let's try a simpler "hill climbing" on the size.
    Since we are greedy, we might get stuck in a local maximum.
    
    Alternative: Randomized Greedy with Backtracking? No, too slow.
    
    Let's try:
    1. Get a greedy family.
    2. For each set NOT in the family, check why it was rejected.
       It was rejected because it formed a sunflower with some pair (i, j) in the family.
    3. If we remove i or j, can we add the new set?
    4. If yes, do the swap (remove i, add new set). Size remains same.
    5. After the swap, try to add more sets greedily.
    
    This is a "1-swap" local search.
    """
    import random
    import time
    
    start_time = time.time()
    family_set = set(family)
    current_size = len(family)
    
    # List of sets not in family
    non_family = [i for i in range(NUM_SETS) if i not in family_set]
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        random.shuffle(non_family)
        
        for new_idx in non_family:
            if time.time() - start_time > time_limit:
                break
                
            # Check if new_idx can be added to current family
            # If yes, add it and continue
            if can_add(new_idx, family):
                family.append(new_idx)
                family_set.add(new_idx)
                non_family.remove(new_idx)
                current_size += 1
                improved = True
                continue
            
            # If not, find the blocking pairs
            # We need to find if removing one set from family allows adding new_idx
            # This is O(m^2) to find the blockers.
            # Then for each blocker, try the swap.
            
            # This might be too slow if done for every non_family member.
            # Let's limit the number of attempts.
            # Or just do random swaps.
            
            # Find a random set in family to remove, and see if new_idx can be added
            # This is a random restart of the local search step.
            
            # Let's pick a random set from family to remove
            if family:
                remove_idx = random.choice(family)
                # Check if new_idx can be added to family \ {remove_idx}
                # This is O(m^2)
                # Let's create a temporary list
                temp_family = [x for x in family if x != remove_idx]
                if can_add(new_idx, temp_family):
                    # Perform swap
                    family.remove(remove_idx)
                    family_set.remove(remove_idx)
                    family.append(new_idx)
                    family_set.add(new_idx)
                    non_family.remove(new_idx)
                    improved = True
                    # Don't break, keep trying to add more
                    break # Restart the loop over non_family? Or continue?
                    # Continuing might be better, but the list is modified.
                    # Let's break to restart the shuffle and loop.
                    # Actually, just breaking the inner loop and continuing the while loop is fine.
        
    return family

def construct():
    start_time = time.time()
    timeout = 80.0 # Safety margin for 90s limit
    
    best_family = []
    best_size = 0
    
    seed_counter = 0
    while time.time() - start_time < timeout:
        # Run greedy randomized
        fam = greedy_randomized(seed=seed_counter)
        
        # Run local search on the result
        fam_ls = local_search(fam, time_limit=1.0)
        
        if len(fam_ls) > best_size:
            best_size = len(fam_ls)
            best_family = fam_ls
            # print(f"New best: {best_size}")
            
        seed_counter += 1
        
        # If we have enough time, do more local search on the best family?
        # Or just keep restarting.
        
    # Convert best_family (list of indices) to list of sorted lists
    result = []
    for idx in best_family:
        result.append(list(ALL_COMBOS[idx]))
        
    return result

if __name__ == "__main__":
    # Quick self-test
    # res = construct()
    # print(f"Size: {len(res)}")
    pass

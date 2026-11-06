import time
import random
from itertools import combinations

# Precomputation
def precompute():
    n = 16
    k = 4
    all_sets = []
    for combo in combinations(range(1, n+1), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    num_sets = len(all_sets)
    # inter[i][j] will store the intersection mask
    # We only need i < j, but for easy access, let's make it a list of lists or a dict
    # A list of lists of size 1820x1820 is ~3.3M entries. Each entry is an int.
    # This is fine.
    inter = [[0]*num_sets for _ in range(num_sets)]
    for i in range(num_sets):
        for j in range(i+1, num_sets):
            inter[i][j] = all_sets[i] & all_sets[j]
            inter[j][i] = inter[i][j]
            
    return all_sets, inter, num_sets

ALL_SETS, INTER, NUM_SETS = precompute()

def can_add(v, family):
    """
    Check if set v can be added to the family.
    family is a list of indices.
    """
    # Check all pairs in family
    # Optimization: if family is large, this is slow.
    for i in range(len(family)):
        fi = family[i]
        for j in range(i+1, len(family)):
            fj = family[j]
            c = INTER[fi][fj]
            if INTER[fi][v] == c and INTER[fj][v] == c:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = 50.0 # 50s limit to be safe
    
    best_family = []
    best_size = 0
    
    # Greedy initialization
    # Try to build a family greedily
    current_family = []
    for v in range(NUM_SETS):
        if can_add(v, current_family):
            current_family.append(v)
            
    best_family = current_family[:]
    best_size = len(current_family)
    
    # Local Search
    # We will perform random swaps and additions
    # To make it faster, we can maintain a list of "blocked" sets? 
    # No, just re-evaluate.
    
    # Shuffle the order of sets to avoid bias
    order = list(range(NUM_SETS))
    random.shuffle(order)
    
    # We will try to add sets from the shuffled order to the current family
    # If we can't add, we try swaps.
    
    # Better strategy: 
    # 1. Try to add a random set not in family.
    # 2. If fail, try a random swap.
    # 3. If fail, try to remove a random set and then add a random set (kick).
    
    family = best_family[:]
    
    while time.time() < deadline:
        # Try to add
        added = False
        for _ in range(10): # Try a few random additions
            v = random.randint(0, NUM_SETS - 1)
            if v not in family:
                if can_add(v, family):
                    family.append(v)
                    added = True
                    if len(family) > best_size:
                        best_size = len(family)
                        best_family = family[:]
                    break
        
        if added:
            continue
            
        # Try to swap
        swapped = False
        for _ in range(10):
            if len(family) == 0:
                break
            u = random.choice(family)
            v = random.randint(0, NUM_SETS - 1)
            if v == u or v in family:
                continue
            
            # Check if family \ {u} + {v} is valid
            # This is equivalent to checking if v can be added to family \ {u}
            # We can do this by temporarily removing u from the check list
            # Or just call a modified can_add
            
            # Optimization: 
            # can_add(v, family) checks pairs in family.
            # We need to check pairs in family \ {u}.
            # Let's just create a temporary list without u? 
            # That's O(N). 
            # Or we can modify can_add to take an "exclude" index.
            
            # Let's write a helper that checks against a list, but skips a specific index?
            # Or simpler: 
            # If can_add(v, family) is False, it's because of some pair.
            # If that pair involves u, then removing u might fix it.
            
            # Let's just do the check properly.
            # Create list without u
            temp_family = [x for x in family if x != u]
            if can_add(v, temp_family):
                family = temp_family
                family.append(v)
                swapped = True
                break
        
        if swapped:
            # Check if we improved? No, size is same.
            # But we might be in a better position.
            continue
        
        # If we couldn't add or swap, maybe do a kick?
        # Remove 1-2 random sets to allow more additions later?
        # This is risky but can escape local optima.
        # Let's just restart with a new random seed if stuck for too long?
        # Or just continue trying random swaps.
        
    return [sorted(ALL_SETS[v].bit_length() and [i+1 for i in range(16) if (ALL_SETS[v] >> i) & 1]) for v in best_family]

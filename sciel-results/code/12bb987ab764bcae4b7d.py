import time
import itertools
import random
import sys

def _to_mask(lst):
    """Convert list of vertices to a 12-bit mask."""
    mask = 0
    for v in lst:
        mask |= (1 << v)
    return mask

def _from_mask(mask):
    """Convert 12-bit mask to list of vertices."""
    return [i for i in range(12) if (mask >> i) & 1]

def _is_comparable(a, b):
    """Check if a and b are comparable (one is subset of other)."""
    if a == b:
        return False
    return (a & b) == a or (a & b) == b

def _count_comparable(S):
    """Count unordered comparable pairs in set S (list of masks)."""
    count = 0
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            if _is_comparable(S[i], S[j]):
                count += 1
    return count

def _local_search(masks, all_masks, deadline):
    """
    Perform local search on the set of masks.
    masks: list of 40 masks
    all_masks: list of all 4096 masks
    deadline: time limit
    """
    S = set(masks)
    current_score = _count_comparable(masks)
    
    # Precompute a list of all masks not in S for faster access?
    # Or just iterate all_masks and check membership.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_x = None
        best_y = None
        
        # Try all swaps
        # For each x in S, for each y not in S
        # Calculate gain of swapping x with y
        
        # Optimization: 
        # gain(x, y) = (comparables of y in S \ {x}) - (comparables of x in S \ {x})
        # Let C(z, S) be count of elements in S comparable to z.
        # Note C(z, S) includes z if z in S? No, z is not comparable to itself in strict subset sense?
        # The problem says "strict subset". 
        # My _is_comparable returns False if a==b.
        # So C(z, S) = sum(1 for w in S if w != z and comparable(z, w)).
        
        # If we remove x:
        # We lose pairs (x, w) for w in S \ {x}. Count = C(x, S).
        # If we add y:
        # We gain pairs (y, w) for w in S \ {x}.
        # Let C_y_S = sum(1 for w in S if w != y and comparable(y, w)).
        # But y is not in S. So C_y_S = sum(1 for w in S if comparable(y, w)).
        # However, if x and y are comparable, the pair (x, y) is NOT in the new set.
        # Wait.
        # Old Score includes pairs among S.
        # New Score includes pairs among S' = S \ {x} + {y}.
        # Pairs in S' are:
        # 1. Pairs among S \ {x}. (Same as before, minus pairs involving x)
        # 2. Pairs involving y and w in S \ {x}.
        
        # Old Score = Score(S \ {x}) + C(x, S \ {x})
        # New Score = Score(S \ {x}) + C(y, S \ {x})
        # Gain = C(y, S \ {x}) - C(x, S \ {x})
        
        # C(x, S \ {x}) is just C(x, S) because x is not comparable to itself.
        # C(y, S \ {x}) = C(y, S) - [x is comparable to y]
        # Because if x ~ y, then x contributed 1 to C(y, S). Since we remove x, we subtract 1.
        # If x !~ y, then C(y, S \ {x}) = C(y, S).
        
        # So Gain(y, x) = C(y, S) - [x ~ y] - C(x, S)
        
        # We can precompute C(x, S) for all x in S.
        # And for each y not in S, compute C(y, S).
        
        # Let's compute C(x, S) for all x in S first.
        cx = {}
        for x in S:
            count = 0
            for w in S:
                if x != w and _is_comparable(x, w):
                    count += 1
            cx[x] = count
        
        # Now iterate y not in S
        for y in all_masks:
            if y in S:
                continue
            
            # Calculate C(y, S)
            cy = 0
            for w in S:
                if _is_comparable(y, w):
                    cy += 1
            
            # For each x in S, calculate gain
            # Gain = cy - [x ~ y] - cx[x]
            # We want to maximize this.
            # [x ~ y] is 1 if x ~ y, else 0.
            
            # We can just loop x again
            for x in S:
                is_comp = 1 if _is_comparable(x, y) else 0
                gain = cy - is_comp - cx[x]
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_y = y
        
        if best_x is not None and best_y is not None:
            # Perform swap
            S.remove(best_x)
            S.add(best_y)
            masks = list(S) # Update list for next iteration
            current_score += best_gain
            improved = True

    return current_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline for safety
    
    # Generate all masks
    all_masks = list(range(4096))
    
    # Heuristic Initialization: Smallest sets
    # Sort by popcount, then by value
    def popcount(x):
        return bin(x).count('1')
        
    sorted_masks = sorted(all_masks, key=lambda x: (popcount(x), x))
    
    best_score = -1
    best_solution = []
    
    # Try a few random restarts if time permits
    # But the "smallest sets" is a very strong candidate.
    # Let's start with that.
    
    initial_masks = sorted_masks[:40]
    
    # Run local search
    score = _local_search(initial_masks, all_masks, deadline)
    
    if score > best_score:
        best_score = score
        best_solution = initial_masks[:]
        
    # If time remains, try random restarts
    if time.time() < deadline - 5:
        random.seed(42)
        for restart in range(10):
            if time.time() > deadline - 2:
                break
            
            # Random initialization
            random.shuffle(all_masks)
            random_masks = all_masks[:40]
            
            score = _local_search(random_masks, all_masks, deadline)
            
            if score > best_score:
                best_score = score
                best_solution = random_masks[:]
                
    # Convert best_solution masks to lists
    result = [_from_mask(m) for m in best_solution]
    
    # Verify validity
    assert len(result) == 40
    seen = set()
    for s in result:
        assert isinstance(s, (list, tuple))
        for x in s:
            assert isinstance(x, int) and not isinstance(x, bool)
            assert 0 <= x <= 11
        fs = frozenset(s)
        assert fs not in seen
        seen.add(fs)
        
    return result

if __name__ == "__main__":
    construct()

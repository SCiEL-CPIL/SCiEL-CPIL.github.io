import time
import random
from itertools import combinations
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline to be safe
    
    N = 128
    # 0-indexed for internal logic, 1-indexed for output
    # Elements are 0 to 127 representing 1 to 128
    
    # Precompute conflicts
    # conflicts[x] is a list of (a, b) such that {x, a, b} is a bad triple.
    # We assume a < b and a, b != x.
    conflicts = [[] for _ in range(N)]
    
    # Helper to check if a triple is bad
    def is_bad(a, b, c):
        # a, b, c are 1-indexed values
        # Check for each prime p up to 128
        # Actually, we can just check the LCM condition directly for small N?
        # lcm(a,b) == lcm(a,c) == lcm(b,c)
        # This is O(1) if we use math.lcm, but might be slow in inner loop?
        # math.lcm is fast in C.
        return lcm(a, b) == lcm(a, c) == lcm(b, c)
        
    # Precomputation
    # Iterate all triples
    # 128 choose 3 is ~ 350k. This is fast.
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if is_bad(a, b, c):
                    # Add to conflicts
                    conflicts[a-1].append((b-1, c-1))
                    conflicts[b-1].append((a-1, c-1))
                    conflicts[c-1].append((a-1, b-1))
                    
    # Local Search
    # State: bitmask of N bits
    # We use a Python int as bitmask
    
    def count_bad_triples(mask):
        # This is O(N^3) if done naively, but we can use conflicts?
        # No, count_bad_triples is only needed for initial scoring or if we don't track incrementally.
        # In the search, we track incrementally.
        # For simplicity in this snippet, let's just use the incremental update in the search loop.
        pass

    # Initial solution: Greedy
    # Sort by some heuristic? Random?
    # Let's start with a random valid set or just empty and add greedily.
    
    def greedy_construct(order):
        mask = 0
        selected = []
        for x in order:
            # Check if x can be added
            # x is 1-indexed in order, so idx = x-1
            idx = x - 1
            valid = True
            for a, b in conflicts[idx]:
                # Check if a and b are in mask
                if ((mask >> a) & 1) and ((mask >> b) & 1):
                    valid = False
                    break
            if valid:
                mask |= (1 << idx)
                selected.append(x)
        return mask, len(selected)
        
    # We will run multiple restarts with different random orders
    best_mask = 0
    best_size = 0
    best_bad_count = 10**9 # We want to minimize bad count, then maximize size
    
    # Actually, let's use a scoring function for the search:
    # Score = 1000 * size - bad_count
    # We track size and bad_count for the current state.
    
    def get_score(size, bad_count):
        return 1000 * size - bad_count
        
    # Local Search Loop
    # We'll do a simple hill climbing / simulated annealing
    # Since N=128, we can try all single-bit flips as moves.
    
    for restart in range(50): # Max 50 restarts
        if time.time() > deadline:
            break
            
        # Initialize with a random greedy set
        # Shuffle 1..128
        order = list(range(1, N + 1))
        random.shuffle(order)
        mask, size = greedy_construct(order)
        bad_count = 0 # Greedy ensures 0 bad triples by construction?
        # Yes, greedy only adds x if no conflict with existing pairs.
        # So bad_count is 0.
        
        # If we start with a valid set, we just try to improve size.
        # But maybe a different starting point allows larger size?
        # Let's try to improve this mask.
        
        current_mask = mask
        current_size = size
        current_bad = 0
        current_score = get_score(current_size, current_bad)
        
        # Local search
        no_improve_count = 0
        for step in range(1000): # Max 1000 steps per restart
            if time.time() > deadline:
                break
                
            best_move_mask = None
            best_move_score = -10**18
            best_move_size = 0
            best_move_bad = 0
            
            # Try all possible single bit flips
            # Adding a bit
            for i in range(N):
                if (current_mask >> i) & 1:
                    continue # Already in set
                # Try adding i
                new_mask = current_mask | (1 << i)
                # Calculate new bad count
                # New bad triples are those involving i and two existing elements
                new_bad = 0
                for a, b in conflicts[i]:
                    if ((current_mask >> a) & 1) and ((current_mask >> b) & 1):
                        new_bad += 1
                new_size = current_size + 1
                new_bad_count = current_bad + new_bad
                new_score = get_score(new_size, new_bad_count)
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_mask = new_mask
                    best_move_size = new_size
                    best_move_bad = new_bad_count
                    
            # Removing a bit
            for i in range(N):
                if not ((current_mask >> i) & 1):
                    continue # Not in set
                # Try removing i
                new_mask = current_mask & ~(1 << i)
                # Calculate change in bad count
                # Bad triples involving i are removed.
                # How many bad triples involve i?
                # We need to count how many pairs (a,b) in conflicts[i] are in current_mask.
                removed_bad = 0
                for a, b in conflicts[i]:
                    if ((current_mask >> a) & 1) and ((current_mask >> b) & 1):
                        removed_bad += 1
                new_size = current_size - 1
                new_bad_count = current_bad - removed_bad
                new_score = get_score(new_size, new_bad_count)
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_mask = new_mask
                    best_move_size = new_size
                    best_move_bad = new_bad_count
            
            if best_move_mask is None:
                break # No moves? Should not happen unless N=0
                
            if best_move_score > current_score:
                current_mask = best_move_mask
                current_size = best_move_size
                current_bad = best_move_bad
                current_score = best_move_score
                no_improve_count = 0
                
                # Update global best if valid
                if current_bad == 0 and current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            else:
                no_improve_count += 1
                # If stuck, maybe do a random move or break
                if no_improve_count > 50:
                    break
                    
    # Fallback: Ensure best_mask is valid
    # If best_mask has bad triples, fix it
    # But greedy start ensures valid, and we only move if score improves.
    # If we move to an invalid state, score might still be higher if size increase outweighs penalty?
    # Score = 1000*size - bad.
    # If we add 1 element, size +1, score +1000.
    # If it creates 500 bad triples, score +500.
    # So we might end up in an invalid state with high score.
    # We must project to valid.
    
    # Project best_mask to a valid set
    final_mask = best_mask
    if final_mask == 0:
        # Should not happen, but return {1}
        return [1]
        
    # Check if final_mask is valid
    # We can just run the greedy removal
    # Convert mask to list
    elements = [i+1 for i in range(N) if (final_mask >> i) & 1]
    
    # If we know best_mask came from a valid path and we only accepted improvements, 
    # it might still be invalid if we accepted a move that increased bad count but size more.
    # Let's verify and fix.
    
    def is_valid_set(elements):
        # Check all triples
        # O(K^3)
        for a, b, c in combinations(elements, 3):
            if is_bad(a, b, c):
                return False
        return True
        
    if not is_valid_set(elements):
        # Greedily remove elements to fix
        # Repeatedly find a bad triple and remove one element from it
        # To be efficient, pick the element that appears in the most bad triples
        remaining = set(elements)
        changed = True
        while changed:
            changed = False
            # Find all bad triples in remaining
            bad_triples = []
            for a, b, c in combinations(remaining, 3):
                if is_bad(a, b, c):
                    bad_triples.append((a, b, c))
            if not bad_triples:
                break
            # Count frequency of each element in bad triples
            freq = {}
            for t in bad_triples:
                for x in t:
                    freq[x] = freq.get(x, 0) + 1
            # Remove element with max frequency
            # If tie, pick random
            max_freq = max(freq.values())
            candidates = [x for x, f in freq.items() if f == max_freq]
            rem = random.choice(candidates)
            remaining.remove(rem)
            changed = True
            
        elements = list(remaining)
        
    if not elements:
        elements = [1]
        
    return elements

# Run construct to test? No, just define it.
# The evaluator calls construct().

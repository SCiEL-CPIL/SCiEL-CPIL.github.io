import time
import random
from itertools import combinations

def _is_valid_add(triple, present, triple_to_k222s, k222_list):
    """Check if adding 'triple' to 'present' creates a complete K(2,2,2)."""
    # Get index of triple
    # We need a fast way to check. Let's use a set of indices.
    # 'present' is a set of indices.
    for k_idx in triple_to_k222s[triple]:
        k_set = k222_list[k_idx]
        # Check if all other triples in this K(2,2,2) are in 'present'
        for other_idx in k_set:
            if other_idx != triple and other_idx not in present:
                break
        else:
            # All others are present, so adding 'triple' completes the K(2,2,2)
            return False
    return True

def _precompute_k222s(n):
    """
    Precompute all K(2,2,2) configurations.
    A K(2,2,2) is defined by 3 disjoint pairs: {a,a'}, {b,b'}, {c,c'}.
    The 8 transversal triples are all combinations choosing one from each pair.
    Returns:
    - triple_to_k222s: list of lists, mapping triple index to list of K(2,2,2) indices it belongs to.
    - k222_list: list of sets (or lists) of triple indices for each K(2,2,2).
    """
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Map points to pairs? No, just iterate over all possible disjoint pairs.
    # Number of ways to choose 3 disjoint pairs from n points.
    # For n=16, this is manageable?
    # Choose 6 points out of 16: C(16,6) = 8008.
    # For each set of 6 points, number of ways to partition into 3 pairs is 15.
    # Total K(2,2,2)s is roughly 8008 * 15 = 120,120.
    # Each K has 8 triples.
    # Total entries in triple_to_k222s: 120,120 * 8 = ~960,960. This is fine.
    
    triple_to_k222s = [[] for _ in all_triples]
    k222_list = []
    
    for points in combinations(range(n), 6):
        # Generate all perfect matchings of these 6 points
        # p0-p5.
        # Pair p0 with p1..p5 (5 choices).
        # Then pair remaining 4.
        # This is a standard recursion or hardcode for 6.
        # Let's just iterate permutations? No.
        # Simple backtracking for matchings of 6 nodes.
        def gen_matchings(pts):
            if not pts:
                yield []
                return
            first = pts[0]
            rest = pts[1:]
            for i in range(len(rest)):
                second = rest[i]
                new_rest = rest[:i] + rest[i+1:]
                for m in gen_matchings(new_rest):
                    yield [(first, second)] + m
        
        for matching in gen_matchings(points):
            # Matching is a list of 3 tuples: (a,a'), (b,b'), (c,c')
            # The 8 triples are formed by choosing one from each pair.
            k_triples = []
            for x in [0, 1]:
                for y in [0, 1]:
                    for z in [0, 1]:
                        t = tuple(sorted((matching[0][x], matching[1][y], matching[2][z])))
                        k_triples.append(t)
            
            k_indices = [triple_to_idx[t] for t in k_triples]
            k222_idx = len(k222_list)
            k222_list.append(set(k_indices))
            
            for idx in k_indices:
                triple_to_k222s[idx].append(k222_idx)
                
    return all_triples, triple_to_k222s, k222_list

def _greedy_seed(all_triples, triple_to_k222s, k222_list, order):
    """
    Greedy construction using a specific order of triples.
    """
    present = set()
    for idx in order:
        if _is_valid_add(idx, present, triple_to_k222s, k222_list):
            present.add(idx)
    return present

def _local_search(initial_set, all_triples, triple_to_k222s, k222_list, time_limit=10.0):
    """
    Local search using swaps to improve the set.
    """
    current_set = set(initial_set)
    best_set = set(current_set)
    best_size = len(current_set)
    start_time = time.time()
    
    # Precompute list of triples not in current_set for random selection
    # And list of triples in current_set
    
    while time.time() - start_time < time_limit:
        current_list = list(current_set)
        if not current_list:
            break
        
        # Pick a random triple to remove
        u_idx = random.choice(current_list)
        
        # Pick a random triple to add (not in current_set)
        # Efficient way: pick a random index from 0..N-1, if it's in current_set, retry or pick next?
        # Since N=560, rejection sampling is fine.
        t_idx = random.randint(0, len(all_triples) - 1)
        while t_idx in current_set:
            t_idx = random.randint(0, len(all_triples) - 1)
        
        # Check if swap is valid
        # 1. Removing u_idx always "un-blocks" things, never creates a new K(2,2,2) violation 
        #    because we are removing a triple. A violation requires 8 triples to be present.
        #    Removing one can only break a violation, not create one.
        # 2. Adding t_idx might create a violation.
        #    We need to check if adding t_idx to (current_set - {u_idx}) creates a K(2,2,2).
        #    Since current_set was valid, any K(2,2,2) containing u_idx was either:
        #    a) Complete (impossible, valid set)
        #    b) Missing >= 2 triples.
        #    If we remove u_idx, K's containing u_idx are now missing u_idx.
        #    If we add t_idx, we need to check K's containing t_idx.
        #    Could a K containing t_idx become complete?
        #    It must have been missing only t_idx in (current_set - {u_idx}).
        #    So in current_set, it was missing t_idx and u_idx (if u_idx was in it) OR just t_idx (if u_idx not in it).
        #    Case 1: u_idx in K. K was missing t_idx and u_idx. After swap, K has t_idx, no u_idx. Still missing u_idx. So not complete.
        #    Case 2: u_idx not in K. K was missing t_idx. After swap, K has t_idx. If all other members are in (current_set - {u_idx}), then K is complete.
        #    Note: if u_idx not in K, then current_set and (current_set - {u_idx}) have the same intersection with K (except t_idx).
        #    So we just need to check if t_idx completes any K(2,2,2) where all other members are in current_set.
        #    Wait, if u_idx is in K, it doesn't matter because u_idx is removed.
        #    So the check is: for all K in triple_to_k222s[t_idx], are all other members of K in current_set?
        #    If yes, then adding t_idx makes K complete?
        #    No! If u_idx is in K, then K is missing u_idx (removed) and has t_idx (added). So K is NOT complete.
        #    So we must verify that u_idx is NOT in any K that would be completed by t_idx.
        
        valid_swap = True
        for k_idx in triple_to_k222s[t_idx]:
            k_set = k222_list[k_idx]
            # Check if all other members of K are in current_set
            # And that u_idx is NOT in k_set (because if it is, K is still missing u_idx)
            is_blocked = True
            for other in k_set:
                if other == t_idx:
                    continue
                if other not in current_set:
                    is_blocked = False
                    break
                if other == u_idx:
                    # If u_idx is in this K, then removing it means K is not complete
                    # even if t_idx is added.
                    # So this K does NOT cause a violation.
                    # But wait, the logic above "is_blocked" is meant to detect if a violation WOULD occur.
                    # A violation occurs if K becomes complete.
                    # K becomes complete if all its members are in the new set.
                    # New set = current_set - {u_idx} + {t_idx}.
                    # If u_idx is in K, then u_idx is NOT in new set. So K is not complete.
                    # So we don't care about K's containing u_idx.
                    # We only care about K's NOT containing u_idx.
                    # If K does not contain u_idx, and all other members (besides t_idx) are in current_set, 
                    # then they are in new set. And t_idx is in new set. So K is complete.
                    # So:
                    if True:
                         # We are in the loop checking "other" in k_set.
                         # If we find u_idx in k_set, this K is safe (won't be complete).
                         # We should break the "is_blocked" check for this K immediately as SAFE?
                         # No, we want to know if this K causes a violation.
                         # If u_idx is in K, it does NOT cause a violation.
                         # So we should mark this K as "not a problem".
                         pass
            # Let's rewrite the check clearly.
            
        # Let's restart the validity check logic.
        valid_swap = True
        for k_idx in triple_to_k222s[t_idx]:
            k_set = k222_list[k_idx]
            # If u_idx is in k_set, this K will NOT be complete after swap (missing u_idx).
            if u_idx in k_set:
                continue
            
            # If u_idx is not in k_set, check if all other members are in current_set.
            all_present = True
            for other in k_set:
                if other != t_idx and other not in current_set:
                    all_present = False
                    break
            if all_present:
                # This K will be complete after swap. Violation!
                valid_swap = False
                break
        
        if valid_swap:
            current_set.remove(u_idx)
            current_set.add(t_idx)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
        else:
            # Do nothing, keep current_set
            pass
            
    return best_set

def construct():
    start_time = time.time()
    deadline = 120.0  # Safe margin from 150s
    
    n = 16
    all_triples, triple_to_k222s, k222_list = _precompute_k222s(n)
    
    best_set = set()
    
    # Run multiple random restarts
    # Each restart:
    # 1. Generate a random order of triples.
    # 2. Greedy construction.
    # 3. Local search.
    
    while time.time() - start_time < deadline - 5.0:  # Leave 5s for safety
        random.seed()  # Reseed for variety
        order = list(range(len(all_triples)))
        random.shuffle(order)
        
        # Greedy
        g_set = _greedy_seed(all_triples, triple_to_k222s, k222_list, order)
        
        if len(g_set) > len(best_set):
            best_set = g_set
            # Optionally run local search immediately on improvement?
            # Or just save it and run local search on best at the end?
            # Local search is expensive. Let's run it on every restart but with short time limit?
            # Or run it on the best set found so far?
            
        # Try local search on the greedy result if there is time
        if time.time() - start_time < deadline - 10.0:
            ls_set = _local_search(g_set, all_triples, triple_to_k222s, k222_list, time_limit=5.0)
            if len(ls_set) > len(best_set):
                best_set = ls_set
        
        # Also try local search on the best_set found so far?
        # Might be too slow. Let's stick to local search on fresh greedy results.
        
    return [list(all_triples[i]) for i in best_set]

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Found {len(res)} triples in {t1-t0:.2f}s")
    # Verify validity
    # (Omitted for brevity in final output, but important for development)

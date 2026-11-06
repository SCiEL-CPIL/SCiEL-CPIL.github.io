"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random

def construct():
    N = 128
    
    # 1. Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # 2. Identify all bad triples in [1, N]
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    # bad_triples_by_element[x] will contain (y, z) if (x, y, z) is bad
    bad_triples_by_element = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                    # Store in all three
                    bad_triples_by_element[a].append((b, c))
                    bad_triples_by_element[b].append((a, c))
                    bad_triples_by_element[c].append((a, b))

    def greedy_remove_and_reinsert(seed):
        random.seed(seed)
        S = set(range(1, N + 1))
        
        # active_count[x] = number of bad triples containing x where all 3 elements are in S
        active_count = [0] * (N + 1)
        
        # Initialize active_count
        # We can just count how many neighbors are in S for each x
        # But initially S is all elements, so active_count[x] = len(bad_triples_by_element[x])
        # Wait, bad_triples_by_element[x] contains (y,z). If y,z in S, it's active.
        # Initially S = [1..N], so all are active.
        for x in range(1, N + 1):
            active_count[x] = len(bad_triples_by_element[x])
        
        removed = []
        
        # Phase 1: Greedy Removal
        while True:
            # Find element with max active_count
            max_count = -1
            victim = -1
            for x in S:
                if active_count[x] > max_count:
                    max_count = active_count[x]
                    victim = x
            
            if victim == -1 or max_count == 0:
                break
            
            # Remove victim
            S.remove(victim)
            removed.append(victim)
            
            # Update active_count for neighbors of victim
            for y, z in bad_triples_by_element[victim]:
                # The triple (victim, y, z) is no longer active because victim is removed.
                # So active_count[y] and active_count[z] decrease by 1.
                # Note: We only care if y and z are still in S? 
                # Actually, active_count[y] counts active triples containing y.
                # The triple (victim, y, z) was active iff y in S and z in S.
                # Since we are removing victim, this triple becomes inactive.
                # So we decrement active_count[y] and active_count[z] IF y and z were in S.
                # But wait, if y or z was already removed, the triple wasn't active anyway.
                # However, our active_count was maintained such that it only counts triples where ALL 3 are in S.
                # So, if y in S and z in S, we decrement.
                if y in S and z in S:
                    active_count[y] -= 1
                    active_count[z] -= 1
                    # active_count[victim] is not used anymore, but for consistency:
                    # active_count[victim] -= 1 (not strictly necessary as victim is out)
        
        # Phase 2: Random Re-insertion
        random.shuffle(removed)
        
        for x in removed:
            # Check if x can be added to S without creating a bad triple
            # A bad triple is created if there exists (y, z) in S such that (x, y, z) is bad.
            valid = True
            for y, z in bad_triples_by_element[x]:
                if y in S and z in S:
                    valid = False
                    break
            
            if valid:
                S.add(x)
                # Update active_count for neighbors? 
                # We don't strictly need to maintain active_count for the re-insertion phase 
                # if we just check validity directly. 
                # But if we wanted to do more swaps, we would. 
                # For now, simple check is O(degree(x)).
        
        return len(S), S

    best_size = 0
    best_set = None
    
    # Run multiple iterations
    # The removal is deterministic based on max count. Ties are broken by iteration order (smallest index first in my loop above? No, I iterate S which is a set, so order is arbitrary but deterministic for a given Python version/hash seed).
    # To introduce variation, we can randomize the tie-breaking or the initial set?
    # Actually, the re-insertion order is randomized.
    # Let's run 100 iterations.
    
    for seed in range(100):
        size, S = greedy_remove_and_reinsert(seed)
        if size > best_size:
            best_size = size
            best_set = S
        
        if best_size == N:
            break

    if best_set is None:
        return [1]
        
    return list(best_set)
# EVOLVE-BLOCK-END

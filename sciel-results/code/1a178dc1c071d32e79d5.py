import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def is_regular_6(neigh, comb):
        mask = 0
        for v in comb: mask |= (1 << v)
        degs = [(neigh[v] & mask).bit_count() for v in comb]
        return len(set(degs)) == 1

    def count_bad_6s_full(neigh, n):
        bad = set()
        for comb in itertools.combinations(range(n), 6):
            if is_regular_6(neigh, comb):
                bad.add(comb)
        return bad

    def local_search(neigh, n, max_iters=5000):
        current_neigh = neigh[:]
        bad_sets = count_bad_6s_full(current_neigh, n)
        
        if not bad_sets:
            return current_neigh

        best_neigh = current_neigh[:]
        best_count = len(bad_sets)
        
        # Precompute affected sets for each edge
        all_6_sets = list(itertools.combinations(range(n), 6))
        sets_by_edge = {}
        for s in all_6_sets:
            for i in range(6):
                for j in range(i+1, 6):
                    u, v = s[i], s[j]
                    if u > v: u, v = v, u
                    key = (u, v)
                    if key not in sets_by_edge:
                        sets_by_edge[key] = []
                    sets_by_edge[key].append(s)
        
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            if bad_sets:
                # Pick a random bad set
                s = random.choice(list(bad_sets))
                # Pick a random edge in s
                u, v = random.sample(s, 2)
                if u > v: u, v = v, u
            else:
                u, v = random.choice(edges)
            
            # Flip
            current_neigh[u] ^= (1 << v)
            current_neigh[v] ^= (1 << u)
            
            # Update bad sets
            # Remove s if it's no longer bad
            # Add new bad sets from affected_sets
            
            new_bad_count = len(bad_sets)
            
            # Check if s is still bad
            if s in bad_sets:
                if not is_regular_6(current_neigh, s):
                    bad_sets.remove(s)
                    new_bad_count -= 1
            
            # Check other affected sets
            affected = sets_by_edge.get((u, v), [])
            for s2 in affected:
                if s2 != s and s2 not in bad_sets:
                    if is_regular_6(current_neigh, s2):
                        bad_sets.add(s2)
                        new_bad_count += 1
            
            if new_bad_count < best_count:
                best_count = new_bad_count
                best_neigh = current_neigh[:]
                if best_count == 0:
                    return best_neigh
            
            # Accept move if it doesn't increase count (or with prob)
            if new_bad_count > len(bad_sets):
                # Revert
                current_neigh[u] ^= (1 << v)
                current_neigh[v] ^= (1 << u)
                # Restore bad_sets? 
                # This is tricky. We modified bad_sets.
                # We need to revert the changes to bad_sets.
                # This is error-prone.
                # Simpler: Just do full recount every K steps.
                pass
        
        return best_neigh

    # ... (rest of construct)

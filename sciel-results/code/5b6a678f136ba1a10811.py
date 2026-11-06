import time
import random
import sys

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    def is_valid_maximal(sidon_set):
        if not sidon_set:
            return False
        # Check Sidon property
        sums = set()
        s_list = list(sidon_set)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i, len(s_list)):
                b = s_list[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        
        # Check Maximal property
        # For every x in [1, N] not in sidon_set, adding x must create a conflict.
        # Conflict if 2x in sums OR (x + y) in sums for some y in sidon_set.
        # Note: if x+y == x+x, it's covered by 2x check.
        # Actually, the new sums are {2x} U {x+y | y in S}.
        # If any of these is in `sums`, it's a conflict.
        
        # Optimization: Precompute sums.
        # For x not in S:
        #   if 2x in sums: conflict
        #   else: check if any x+y in sums.
        #   Note: x+y in sums means there exist a,b in S such that a+b = x+y.
        #   Since x not in S, we are checking if x = a+b-y.
        
        # Let's just iterate. N=256 is small.
        for x in range(1, N + 1):
            if x in sidon_set:
                continue
            if 2 * x in sums:
                continue
            conflict = False
            for y in s_list:
                if x + y in sums:
                    conflict = True
                    break
            if not conflict:
                return False
        return True

    def get_conflicts_for_removal(sidon_set):
        """
        Returns a list of elements that can be removed while keeping the set maximal Sidon.
        """
        if not is_valid_maximal(sidon_set):
            return []
        
        removable = []
        s_list = list(sidon_set)
        # Precompute sums for current set
        sums = set()
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i, len(s_list)):
                sums.add(a + s_list[j])
                
        for elem in s_list:
            # Try removing elem
            new_set = set(sidon_set)
            new_set.discard(elem)
            
            # Check if new_set is Sidon
            # Since original was Sidon, removing an element preserves Sidon property.
            # We only need to check maximality.
            
            # However, we need the sums of new_set to check maximality.
            # Recalculating sums for every removal is O(|S|^2) * |S| = O(|S|^3).
            # |S| ~ 16. 16^3 = 4096. Very fast.
            
            # Let's optimize: 
            # Sums of new_set = Sums of old_set - {a+elem | a in old_set, a<=elem? No, a+b where one is elem}
            # It's easier to just rebuild sums for the check, or maintain a count.
            # Given N=256, rebuilding is fine.
            
            new_sums = set()
            new_list = list(new_set)
            valid_sidon = True
            for i in range(len(new_list)):
                a = new_list[i]
                for j in range(i, len(new_list)):
                    s = a + new_list[j]
                    if s in new_sums:
                        valid_sidon = False
                        break
                    new_sums.add(s)
            if not valid_sidon:
                continue
                
            # Check maximality of new_set
            maximal = True
            for x in range(1, N + 1):
                if x in new_set:
                    continue
                if 2 * x in new_sums:
                    continue
                conflict = False
                for y in new_list:
                    if x + y in new_sums:
                        conflict = True
                        break
                if not conflict:
                    maximal = False
                    break
            
            if maximal:
                removable.append(elem)
                
        return removable

    # Initial Solution: Greedy Ascending
    def greedy_construct():
        selected = []
        sums = set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return set(selected)

    best_set = greedy_construct()
    best_size = len(best_set)
    
    # Local Search
    while time.time() < deadline:
        # Try to remove elements
        removed_any = False
        # Shuffle to break symmetry and explore different removal orders
        removable = get_conflicts_for_removal(best_set)
        random.shuffle(removable)
        
        for elem in removable:
            if time.time() > deadline:
                break
            # Verify again in case previous removals in this iteration affected it?
            # The function `get_conflicts_for_removal` checks each independently against the *current* best_set.
            # If we remove one, the set changes, so subsequent elements in `removable` might no longer be removable.
            # So we should re-evaluate or just try to remove one, check validity, and if valid, accept and restart loop.
            
            new_set = set(best_set)
            new_set.discard(elem)
            
            if is_valid_maximal(new_set):
                best_set = new_set
                best_size = len(best_set)
                removed_any = True
                break # Restart loop with new best_set
        
        if not removed_any:
            # Stuck at a local minimum. Try random restarts or perturbations?
            # Perturbation: Remove a random element (even if it breaks maximality? No, must stay valid).
            # We can try "kick" moves: remove one element, then add a different one?
            # Simpler: Random restart with different random seeds for construction?
            # Greedy is deterministic. Let's try a randomized greedy.
            
            # Randomized Greedy
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            selected = []
            sums = set()
            for x in candidates:
                added = [x + a for a in selected] + [2 * x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    selected.append(x)
                    sums.update(added)
            
            rand_set = set(selected)
            if len(rand_set) < best_size:
                best_set = rand_set
                best_size = len(best_set)
            else:
                # If same size or worse, we might still try to improve this specific random set
                # But to save time, only pursue if promising.
                if len(rand_set) <= best_size:
                    best_set = rand_set
                    # Continue local search on this new set
                    continue
                else:
                    # If we are stuck and random restarts aren't helping, we might just be at the global minimum for this heuristic.
                    # We can try a few more random restarts.
                    pass

    return list(best_set)
# EVOLVE-BLOCK-END

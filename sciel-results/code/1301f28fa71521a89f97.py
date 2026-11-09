import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50.0
    T = 200
    N = 100
    mask_all = (1 << (T + 1)) - 1
    
    def get_reachable(selected):
        """Calculate reachable sums bitmask for a given list of selected elements."""
        reachable = 1
        for x in selected:
            reachable |= (reachable << x)
            reachable &= mask_all
        return reachable

    def is_valid(reachable):
        return (reachable >> T) & 1 == 0

    # 1. Initial Greedy Construction
    # Try to build a large valid set greedily
    def greedy_init(order):
        selected = []
        reachable = 1
        for x in order:
            if x > T:
                continue
            # Check if adding x creates sum T
            if x <= T and ((reachable >> (T - x)) & 1):
                continue
            selected.append(x)
            reachable |= (reachable << x)
            reachable &= mask_all
        return selected, reachable

    # Start with a greedy set
    best_set, best_reach = greedy_init(range(N, 0, -1))
    best_size = len(best_set)
    
    # Local improvement on the greedy set
    def local_improve(selected, reachable):
        sset = set(selected)
        excluded = [x for x in range(N, 0, -1) if x not in sset]
        changed = True
        while changed:
            changed = False
            for x in excluded:
                if x in sset:
                    continue
                if x <= T and ((reachable >> (T - x)) & 1):
                    continue
                selected.append(x)
                sset.add(x)
                reachable |= (reachable << x)
                reachable &= mask_all
                excluded.remove(x)
                changed = True
                if time.time() > deadline:
                    return selected, reachable
        return selected, reachable

    best_set, best_reach = local_improve(best_set, best_reach)
    best_size = len(best_set)
    
    # 2. Simulated Annealing / Random Walk
    # State: current_set (list), current_reach (int)
    current_set = best_set.copy()
    current_reach = best_reach
    current_size = best_size
    
    # Helper to update reach when adding/removing is complex, 
    # but for N=100, recalculating from scratch on removal is cheap.
    # Adding can be done incrementally.
    
    while time.time() < deadline:
        # Try to add a random element
        if current_size < N:
            sset = set(current_set)
            excluded = [x for x in range(1, N + 1) if x not in sset]
            if excluded:
                x = random.choice(excluded)
                # Check if x can be added
                if x <= T and ((current_reach >> (T - x)) & 1) == 0:
                    # Safe to add
                    current_set.append(x)
                    current_reach |= (current_reach << x)
                    current_reach &= mask_all
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set.copy()
                        best_reach = current_reach
                else:
                    # Cannot add x. 
                    # With some probability, "kick" by removing a random element from current_set
                    # to potentially unblock future additions
                    if current_size > 0 and random.random() < 0.5:
                        # Remove a random element
                        idx = random.randint(0, current_size - 1)
                        removed_val = current_set[idx]
                        current_set.pop(idx)
                        current_size -= 1
                        # Recalculate reach from scratch
                        current_reach = get_reachable(current_set)
                        # Try to add the excluded element again or just continue
                        # We don't immediately add x back, we let the loop handle it
                        # But we can try to add x now if it was the blocker? 
                        # Actually, removing an element might make x valid.
                        if x <= T and ((current_reach >> (T - x)) & 1) == 0:
                            current_set.append(x)
                            current_reach |= (current_reach << x)
                            current_reach &= mask_all
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_set = current_set.copy()
                                best_reach = current_reach
        else:
            # Set is full (unlikely to be valid and full)
            break

    # 3. Final Polish: Local Improvement on best_set found
    if best_set:
        best_set, best_reach = local_improve(best_set, best_reach)
        if len(best_set) > best_size:
            best_size = len(best_set)
            best_set = best_set # already updated

    if not best_set:
        best_set = [100]
        
    return best_set
# EVOLVE-BLOCK-END

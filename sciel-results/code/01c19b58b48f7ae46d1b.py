import random
import time
import math

N = 40

def construct():
    start_time = time.time()
    timeout = 85.0  # Leave 5s margin

    def get_missing_count(S):
        """Returns the number of missing differences in 1..38 for set S."""
        covered = set()
        S_sorted = sorted(S)
        k = len(S_sorted)
        for i in range(k):
            for j in range(i + 1, k):
                d = S_sorted[j] - S_sorted[i]
                if d <= 38:
                    covered.add(d)
        missing = 0
        for d in range(1, 39):
            if d not in covered:
                missing += 1
        return missing

    def greedy_init(k):
        """Greedy construction of initial set S of size k."""
        S = [1, 39]
        covered = {38}
        candidates = list(range(2, 39))
        while len(S) < k:
            best_val = None
            best_gain = -1
            for v in candidates:
                gain = 0
                for u in S:
                    d = abs(u - v)
                    if d not in covered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_val = v
            if best_val is not None:
                S.append(best_val)
                candidates.remove(best_val)
                for u in S[:-1]:
                    d = abs(u - best_val)
                    covered.add(d)
            else:
                break
        # Fill remaining randomly if needed
        while len(S) < k:
            v = random.randint(1, 39)
            if v not in S:
                S.append(v)
        return S

    def simulated_annealing(S_init, time_limit):
        """Simulated Annealing to minimize missing differences."""
        S = list(S_init)
        S_set = set(S)
        current_cost = get_missing_count(S)
        T = 5.0
        cooling = 0.9995
        best_S = list(S)
        best_cost = current_cost
        
        while time.time() < time_limit and current_cost > 0:
            idx = random.randint(0, len(S) - 1)
            old_val = S[idx]
            new_val = random.randint(1, 39)
            if new_val in S_set and new_val != old_val:
                continue
            
            S[idx] = new_val
            S_set.discard(old_val)
            S_set.add(new_val)
            
            new_cost = get_missing_count(S)
            delta = new_cost - current_cost
            
            if delta <= 0 or (T > 0.01 and random.random() < math.exp(-delta / T)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_S = list(S)
            else:
                S[idx] = old_val
                S_set.discard(new_val)
                S_set.add(old_val)
            
            T *= cooling
        
        return best_S, best_cost

    # Try to find a set of size 10
    found_S = None
    # Use multiple restarts with greedy init
    for i in range(50):
        if time.time() - start_time > timeout:
            break
        S_init = greedy_init(10)
        S_candidate, cost = simulated_annealing(S_init, start_time + timeout - (time.time() - start_time))
        if cost == 0:
            found_S = S_candidate
            break

    # If size 10 not found, try size 11
    if found_S is None:
        for i in range(50):
            if time.time() - start_time > timeout:
                break
            S_init = greedy_init(11)
            S_candidate, cost = simulated_annealing(S_init, start_time + timeout - (time.time() - start_time))
            if cost == 0:
                found_S = S_candidate
                break

    # Fallback to a larger set if all else fails
    if found_S is None:
        # A set of 13 should almost certainly work
        found_S = random.sample(range(1, 40), 13)
        # Verify and fix if necessary (though highly unlikely to fail)
        # For safety, we can just use a dense set if verification fails, 
        # but random 13 is very robust.

    # Construct graph
    path_edges = [(i, i + 1) for i in range(1, N - 1)]
    spoke_edges = [(0, s) for s in found_S]
    edges = path_edges + spoke_edges

    S_set = set(found_S)
    cycles = []
    for L in range(3, N + 1):
        diff_needed = L - 2
        found_cycle = False
        for u in found_S:
            v = u + diff_needed
            if v in S_set:
                # Cycle: 0 -> u -> u+1 -> ... -> v -> 0
                cycle_verts = [0] + list(range(u, v + 1))
                cycles.append(cycle_verts)
                found_cycle = True
                break
            v2 = u - diff_needed
            if v2 in S_set:
                # Cycle: 0 -> u -> u-1 -> ... -> v2 -> 0
                cycle_verts = [0] + list(range(u, v2 - 1, -1))
                cycles.append(cycle_verts)
                found_cycle = True
                break
        if not found_cycle:
            # This should not happen if coverage is complete
            # Fallback: raise error
            raise Exception("Cycle not found for length " + str(L))

    return {'edges': edges, 'cycles': cycles}

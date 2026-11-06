import random
import time

def construct():
    n = 128
    target_start = 128
    target_end = 256
    
    def build_counts(A_set):
        counts = [0] * (2 * n + 1)
        for a in A_set:
            for b in A_set:
                if a <= b:
                    counts[a + b] += 1
        return counts

    def is_valid(counts):
        for s in range(target_start, target_end + 1):
            if counts[s] == 0:
                return False
        return True

    def get_peak(counts):
        return max(counts[target_start:target_end + 1])

    # Phase 1: Greedy Construction
    # Start with mandatory element 128 (for sum 256)
    A = {128}
    counts = build_counts(A)
    
    # Precompute pairs for each sum for quick lookup
    pairs_for_sum = {}
    for s in range(2 * n + 1):
        pairs_for_sum[s] = []
        for a in range(n + 1):
            b = s - a
            if a <= b and b <= n:
                pairs_for_sum[s].append((a, b))
                
    # Greedy loop to cover all sums
    # We pick the "hardest" uncovered sum to resolve next
    # Hardness: fewest connections to existing A
    
    while time.time() < 5.0: # Time budget for construction
        # Find uncovered sums
        uncovered = [s for s in range(target_start, target_end + 1) if counts[s] == 0]
        if not uncovered:
            break
            
        # Score each uncovered sum by "difficulty"
        # Difficulty: number of pairs (a,b) for s where at least one element is already in A
        # Lower count = harder to cover without adding new elements that might not help other sums?
        # Actually, if 0 pairs have elements in A, we must add a new element from scratch.
        # If some pairs have elements in A, we are "closer".
        # Let's prioritize sums with FEWER connections to existing A to force structure?
        # Or prioritize sums with MORE connections to quickly cover?
        # Let's try processing the sum with the MAXIMUM connections first (easiest to cover)
        # to quickly reduce the number of uncovered sums.
        
        best_s = None
        best_score = -1
        
        for s in uncovered:
            conn_count = 0
            for a, b in pairs_for_sum[s]:
                if a in A or b in A:
                    conn_count += 1
            
            if conn_count > best_score:
                best_score = conn_count
                best_s = s
                
        if best_s is None:
            # All have 0 connections, pick random
            best_s = random.choice(uncovered)
            
        s = best_s
        # Find a pair to add
        # Prefer pairs where one element is already in A
        candidates = []
        for a, b in pairs_for_sum[s]:
            if a in A and b in A:
                # Should not happen if counts[s]==0
                continue
            if a in A:
                # Must add b
                candidates.append((b, 1)) # cost 1 element
            elif b in A:
                # Must add a
                candidates.append((a, 1))
            else:
                # Can add a or b
                # Heuristic: pick the one with lower "degree" in current counts?
                # Or just pick one.
                candidates.append((a, 2))
                candidates.append((b, 2))
                
        if not candidates:
            # Should not happen
            break
            
        # Sort by cost, then random tie-break
        random.shuffle(candidates)
        candidates.sort(key=lambda x: x[1])
        
        # Pick the best candidate
        elem_to_add = candidates[0][0]
        
        # Add element
        A.add(elem_to_add)
        
        # Update counts
        # The new pairs are (elem_to_add, y) for all y in A (which now includes elem_to_add)
        for y in A:
            counts[elem_to_add + y] += 1

    # Ensure we have a valid set. If construction failed (timeout), fallback to dense set
    if not is_valid(counts):
        A = set(range(64, 129))
        counts = build_counts(A)

    # Phase 2: Local Search Optimization
    current_peak = get_peak(counts)
    start_time = time.time()
    timeout = 55.0
    
    while time.time() - start_time < timeout:
        improved = False
        A_list = list(A)
        random.shuffle(A_list)
        
        for a in A_list:
            if time.time() - start_time > timeout:
                break
            
            # Tentatively remove a
            A.remove(a)
            
            # Update counts: subtract pairs involving a
            # Pairs are (a, b) for b in A (current, without a) AND (a, a)
            for b in A:
                counts[a + b] -= 1
            counts[2 * a] -= 1
            
            if is_valid(counts):
                new_peak = get_peak(counts)
                if new_peak <= current_peak:
                    current_peak = new_peak
                    improved = True
                    continue
                else:
                    A.add(a)
                    for b in A:
                        counts[a + b] += 1
                    counts[2 * a] += 1
            else:
                A.add(a)
                for b in A:
                    counts[a + b] += 1
                counts[2 * a] += 1
        
        if not improved:
            # Try swaps
            swapped = False
            for _ in range(50):
                if time.time() - start_time > timeout:
                    break
                if len(A) < 1:
                    break
                a = random.choice(list(A))
                candidates = [x for x in range(n + 1) if x not in A]
                if not candidates:
                    break
                b = random.choice(candidates)
                
                # Remove a
                A.remove(a)
                for c in A:
                    counts[a + c] -= 1
                counts[2 * a] -= 1
                
                # Add b
                for c in A:
                    counts[b + c] += 1
                counts[2 * b] += 1
                A.add(b)
                
                if is_valid(counts):
                    new_peak = get_peak(counts)
                    if new_peak < current_peak:
                        current_peak = new_peak
                        swapped = True
                        break
                    else:
                        # Restore
                        A.remove(b)
                        for c in A:
                            counts[b + c] -= 1
                        counts[2 * b] -= 1
                        
                        A.add(a)
                        for c in A:
                            counts[a + c] += 1
                        counts[2 * a] += 1
                else:
                    # Restore
                    A.remove(b)
                    for c in A:
                        counts[b + c] -= 1
                    counts[2 * b] -= 1
                    
                    A.add(a)
                    for c in A:
                        counts[a + c] += 1
                    counts[2 * a] += 1
            
            if not swapped and not improved:
                # Try a random restart with a slightly different seed?
                # Or just break.
                break
                
    return list(A)

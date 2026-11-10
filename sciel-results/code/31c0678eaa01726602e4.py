import random
import math

def construct():
    n = 256
    
    # Precompute all 4-APs and 3-APs involving each element
    four_ap_partners = {i: [] for i in range(1, n+1)}
    three_ap_partners = {i: [] for i in range(1, n+1)}
    
    # Generate all 4-APs
    for a in range(1, n+1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > n:
                break
            # Add to partners
            four_ap_partners[a].append((b, c, e))
            four_ap_partners[b].append((a, c, e))
            four_ap_partners[c].append((a, b, e))
            four_ap_partners[e].append((a, b, c))
            
    # Generate all 3-APs
    for a in range(1, n+1):
        for d in range(1, (n - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c > n:
                break
            three_ap_partners[a].append((b, c))
            three_ap_partners[b].append((a, c))
            three_ap_partners[c].append((a, b))

    def is_valid_addition(S, x):
        """Check if adding x to S creates a 4-AP."""
        for o1, o2, o3 in four_ap_partners[x]:
            if o1 in S and o2 in S and o3 in S:
                return False
        return True

    def count_3aps_involving(S, x):
        """Count 3-APs in S that involve x."""
        count = 0
        for o1, o2 in three_ap_partners[x]:
            if o1 in S and o2 in S:
                count += 1
        return count

    def total_3aps(S):
        """Total number of 3-APs in S."""
        s = 0
        for x in S:
            s += count_3aps_involving(S, x)
        return s // 3

    # Initialize S
    S = set(range(1, n + 1))
    changed = True
    while changed:
        changed = False
        for a in range(1, n + 1):
            for d in range(1, (n - a) // 3 + 1):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                if e > n:
                    break
                if a in S and b in S and c in S and e in S:
                    candidates = [a, b, c, e]
                    rem = random.choice(candidates)
                    S.remove(rem)
                    changed = True
                    break
            if changed:
                break

    best_S = set(S)
    best_score = total_3aps(S)
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    iterations = 50000
    
    current_score = best_score
    current_S = set(S)
    
    for _ in range(iterations):
        if T < T_min:
            break
            
        if random.random() < 0.5 and len(current_S) < n:
            candidates_to_add = [x for x in range(1, n + 1) if x not in current_S]
            if not candidates_to_add:
                continue
            x = random.choice(candidates_to_add)
            if not is_valid_addition(current_S, x):
                continue
            delta = count_3aps_involving(current_S, x)
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_S.add(x)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(current_S)
        else:
            if not current_S:
                continue
            x = random.choice(list(current_S))
            delta = -count_3aps_involving(current_S, x)
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_S.remove(x)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(current_S)
        
        T *= cooling_rate

    return sorted(best_S)

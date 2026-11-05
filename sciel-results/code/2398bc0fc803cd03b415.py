"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    deadline = time.time() + 85.0
    N = 256
    
    # Precompute all 3-term APs in [1, N]
    # A 3-AP is (a, b, c) with a < b < c and a + c = 2b, i.e., b - a = c - b = d > 0
    # We'll represent each AP as a frozenset of 3 elements
    all_aps = []
    for a in range(1, N + 1):
        for d in range(1, N // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                all_aps.append((a, b, c))
    
    # For fast checking: for each element x, list of APs containing x
    ap_index = {}
    for i, (a, b, c) in enumerate(all_aps):
        for x in (a, b, c):
            if x not in ap_index:
                ap_index[x] = []
            ap_index[x].append(i)
    
    def is_valid(S):
        """Check if set S contains no 3-AP."""
        s = set(S)
        for a, b, c in all_aps:
            if a in s and b in s and c in s:
                return False
        return True
    
    def can_add(S, x):
        """Check if adding x to S creates a 3-AP."""
        if x in S:
            return False
        for ap_idx in ap_index.get(x, []):
            a, b, c = all_aps[ap_idx]
            others = [y for y in (a, b, c) if y != x]
            if all(y in S for y in others):
                return False
        return True
    
    def recip_sum(S):
        total = Fraction(0, 1)
        for x in S:
            total += Fraction(1, x)
        return total
    
    # Greedy initialization: add elements in order of decreasing 1/x (i.e., increasing x)
    # that don't create a 3-AP
    S = set()
    for x in range(1, N + 1):
        if can_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_score = recip_sum(best_S)
    
    # Local search: swap-based improvements
    def local_search(S, score, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = sorted(S)
            for a in S_list:
                if time.time() > deadline - 1:
                    break
                for x in range(1, N + 1):
                    if x in S:
                        continue
                    # Try removing a and adding x
                    trial = S - {a}
                    if can_add(trial, x):
                        new_S = trial | {x}
                        new_score = score - Fraction(1, a) + Fraction(1, x)
                        if new_score > score:
                            S = new_S
                            score = new_score
                            improved = True
                            break
            # Also try pure additions
            for x in range(1, N + 1):
                if time.time() > deadline - 1:
                    break
                if x not in S:
                    if can_add(S, x):
                        S = S | {x}
                        score = score + Fraction(1, x)
                        improved = True
            # Try removals that might enable better additions
            for a in list(S):
                if time.time() > deadline - 1:
                    break
                trial = S - {a}
                trial_score = score - Fraction(1, a)
                # After removing a, try to add back as many as possible
                for x in range(1, N + 1):
                    if x in trial:
                        continue
                    if can_add(trial, x):
                        trial = trial | {x}
                        trial_score = trial_score + Fraction(1, x)
                if trial_score > score:
                    S = trial
                    score = trial_score
                    improved = True
        return S, score
    
    best_S, best_score = local_search(best_S, best_score, deadline)
    
    # Random restarts with perturbation
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        # Create a perturbed version of best_S
        S = set(best_S)
        # Remove 1-3 random elements
        k = random.randint(1, min(3, len(S)))
        to_remove = random.sample(list(S), k)
        for a in to_remove:
            S.discard(a)
        
        # Try to fill back up
        for x in range(1, N + 1):
            if x not in S and can_add(S, x):
                S.add(x)
        
        # Local search from this perturbed state
        S2, score2 = local_search(S, recip_sum(S), min(time.time() + 10, deadline - 2))
        
        if score2 > best_score:
            best_S = S2
            best_score = score2
    
    # Final polish
    best_S, best_score = local_search(best_S, best_score, deadline)
    
    # Verify validity
    if not is_valid(best_S):
        # Fallback: greedy from scratch
        S = set()
        for x in range(1, N + 1):
            if can_add(S, x):
                S.add(x)
        best_S = S
        best_score = recip_sum(best_S)
    
    return sorted(best_S)
# EVOLVE-BLOCK-END

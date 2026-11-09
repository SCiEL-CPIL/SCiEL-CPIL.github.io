"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _check_pair(r1, m1, r2, m2):
    """Return True if classes are disjoint (gcd(m1,m2) does not divide r1-r2)."""
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0

def _is_valid_set(classes):
    """Check if all pairs are disjoint."""
    n = len(classes)
    for i in range(n):
        for j in range(i+1, n):
            r1, m1 = classes[i]
            r2, m2 = classes[j]
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def _find_max_classes(max_mod=128, deadline=None):
    """Backtracking search for maximum pairwise-disjoint residue classes."""
    if deadline is None:
        deadline = time.time() + 80  # safety margin
    
    best = []
    
    # We need to select moduli and assign residues.
    # Key insight: if two moduli are coprime, they can NEVER be in a valid set.
    # So we need a subset of {2,...,max_mod} with no two coprime.
    
    # For efficiency, let's try specific structures:
    # 1. Powers of 2: 2,4,8,16,32,64,128 (7 moduli)
    # 2. All multiples of a prime p
    # 3. Mixed structures
    
    # Try powers of 2 with optimized residues
    # For modulus 2^k, choose residue r_k such that for all pairs,
    # gcd(2^i, 2^j) = 2^min(i,j) does not divide (r_i - r_j)
    
    # Strategy for powers of 2: set r_k = 2^k - 1 (the seed approach)
    # This gives: r_i - r_j = 2^i - 2^j for i<j, = 2^i(1-2^(j-i))
    # gcd = 2^i, and 2^i divides 2^i(1-2^(j-i)), so they'd intersect!
    # Wait, but the seed works... let me recheck.
    # Seed: k in range(7), r = (1<<k)-1, m = 1<<(k+1)
    # k=0: r=0, m=2
    # k=1: r=1, m=4
    # k=2: r=3, m=8
    # k=3: r=7, m=16
    # k=4: r=15, m=32
    # k=5: r=31, m=64
    # k=6: r=63, m=128
    
    # Pair (0,1): gcd(2,4)=2, r0-r1 = -1, 2 ∤ -1. OK
    # Pair (1,2): gcd(4,8)=4, r1-r2 = -2, 4 ∤ -2. OK
    # Pair (2,3): gcd(8,16)=8, r2-r3 = -4, 8 ∤ -4. OK
    # So r_k = 2^k - 1 with m_k = 2^(k+1) works!
    
    # The key: r_k - r_j = (2^k-1)-(2^j-1) = 2^k - 2^j = 2^k(1 - 2^(j-k))
    # For k<j: this is 2^k * (1 - 2^(j-k)). Since (1-2^(j-k)) is odd,
    # the 2-adic valuation is exactly k. gcd = 2^k. So 2^k divides it!
    # But we need gcd NOT to divide it. Contradiction?
    
    # Wait: 2^k(1-2^(j-k)) is divisible by 2^k. So gcd | (r_k - r_j). They intersect!
    # But my manual check showed they don't. Let me recheck pair (1,2):
    # r_1 = 1, r_2 = 3. r_1 - r_2 = -2. gcd(4,8) = 4. 4 ∤ -2. Correct, disjoint.
    # My formula: 2^1 * (1 - 2^1) = 2 * (-1) = -2. Divisible by 2^1=2, yes.
    # But gcd is 4=2^2, not 2^1. I confused the indices.
    
    # m_k = 2^(k+1). So m_1 = 4 = 2^2, m_2 = 8 = 2^3.
    # gcd(m_1, m_2) = 4 = 2^2.
    # r_1 - r_2 = 1 - 3 = -2. 4 ∤ -2. Good.
    
    # General: gcd(m_k, m_j) = 2^min(k+1, j+1) = 2^(k+1) for k<j.
    # r_k - r_j = (2^k-1) - (2^j-1) = 2^k - 2^j = 2^k(1-2^(j-k))
    # v2(r_k - r_j) = k (since 1-2^(j-k) is odd)
    # Need k >= k+1? No, k < k+1. So v2(r_k-r_j) = k < k+1 = v2(gcd).
    # So gcd does NOT divide (r_k - r_j). 
    
    # Great, so powers of 2 with r_k = 2^k-1 always work. 7 classes.
    
    # Can we add more? Try adding modulus 3? gcd(3, 2^k)=1 for all k. Coprime! Can't.
    # Try modulus 6? gcd(6,2)=2, gcd(6,4)=2, gcd(6,8)=2, etc.
    # Need to find r_6 such that for each existing class, gcd(6, m_k) ∤ (r_6 - r_k).
    # gcd(6,2)=2: need 2 ∤ (r_6 - 0) → r_6 odd
    # gcd(6,4)=2: need 2 ∤ (r_6 - 1) → r_6 even (since r_6-1 odd means r_6 even)
    # Contradiction! r_6 can't be both odd and even.
    
    # Try modulus 10? gcd(10,2)=2, gcd(10,4)=2, gcd(10,8)=2, gcd(10,16)=2, etc.
    # Same issue: gcd with 2 is 2, need r_10 odd (vs r_0=0)
    # gcd with 4 is 2, need r_10 even (vs r_1=1)
    # Contradiction again.
    
    # It seems hard to add non-power-of-2 moduli. Let's try a systematic search.
    
    # Approach: try to build a set greedily/backtracking.
    # Since the problem is small enough, we can try all subsets of "candidate" moduli.
    
    # Candidate moduli: those that share a factor with most others.
    # Let's try: all even numbers, and see if we can assign residues.
    
    # For a set of moduli, we need to find residues r_m for each m such that
    # for all pairs (m1, m2): gcd(m1,m2) ∤ (r_{m1} - r_{m2})
    
    # This is a constraint satisfaction problem. Let's try small sets first.
    
    # Start with powers of 2 (7 classes) and try to add more.
    base_classes = [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    best_set = list(base_classes)
    
    # Try to add moduli one by one
    used_moduli = set(m for _, m in best_set)
    candidate_moduli = [m for m in range(2, 129) if m not in used_moduli]
    
    for m_new in candidate_moduli:
        if time.time() > deadline:
            break
        # Check if m_new is coprime with any used modulus
        if any(gcd(m_new, m_old) == 1 for _, m_old in best_set):
            continue
        # Try to find a residue r_new that works
        found = False
        for r_new in range(m_new):
            if time.time() > deadline:
                break
            ok = True
            for r_old, m_old in best_set:
                g = gcd(m_new, m_old)
                if (r_new - r_old) % g == 0:
                    ok = False
                    break
            if ok:
                best_set.append([r_new, m_new])
                used_moduli.add(m_new)
                found = True
                break
        if time.time() > deadline:
            break
    
    # Verify the result
    if _is_valid_set(best_set):
        return best_set
    else:
        # Fallback to seed
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]

def construct():
    result = _find_max_classes(max_mod=128)
    return result
# EVOLVE-BLOCK-END

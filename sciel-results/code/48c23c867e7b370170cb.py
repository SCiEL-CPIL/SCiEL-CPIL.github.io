import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_mods = len(moduli)
    
    best_set = []
    
    # Precompute gcds
    gcds = {}
    for i in range(2, max_mod + 1):
        for j in range(i, max_mod + 1):
            gcds[(i, j)] = gcd(i, j)
            
    def get_gcd(m1, m2):
        if m1 > m2:
            m1, m2 = m2, m1
        return gcds.get((m1, m2), 1)

    def greedy_random(seed_val):
        random.seed(seed_val)
        # blocked[m][r] counts how many chosen pairs conflict with (r, m)
        # We use a list of lists of ints
        blocked = [[0] * m for m in moduli]
        # avail_moduli is a set of indices into moduli list? Or just values?
        # Let's use indices for faster access
        avail_indices = set(range(n_mods))
        chosen_pairs = [] # list of (r, m)
        
        while avail_indices:
            # Find valid residues for each available modulus
            # valid_counts[m_idx] = count of r with blocked[m_idx][r] == 0
            # valid_rs[m_idx] = list of such r
            min_count = float('inf')
            min_candidates = []
            
            for idx in avail_indices:
                m = moduli[idx]
                b_list = blocked[idx]
                # Count zeros
                cnt = 0
                # We can optimize by keeping a list of valid rs, but counting is O(m)
                # Since m <= 128, this is fine.
                for r in range(m):
                    if b_list[r] == 0:
                        cnt += 1
                if cnt > 0:
                    if cnt < min_count:
                        min_count = cnt
                        min_candidates = [idx]
                    elif cnt == min_count:
                        min_candidates.append(idx)
            
            if not min_candidates:
                break
                
            # Pick a random candidate from min_candidates
            pick_idx = random.choice(min_candidates)
            m_pick = moduli[pick_idx]
            
            # Pick a random valid residue
            valid_rs = [r for r in range(m_pick) if blocked[pick_idx][r] == 0]
            r_pick = random.choice(valid_rs)
            
            chosen_pairs.append((r_pick, m_pick))
            avail_indices.remove(pick_idx)
            
            # Update blocked for all remaining moduli
            for idx in avail_indices:
                m_rem = moduli[idx]
                g = get_gcd(m_pick, m_rem)
                # If g == 1, no conflict possible, skip?
                # (r - r_pick) % 1 == 0 is always true.
                # Wait. Conflict if (r - r_pick) % g == 0.
                # If g=1, (r - r_pick) % 1 == 0 is always true.
                # So if g=1, ALL residues conflict?
                # Let's re-verify the condition.
                # Disjoint if gcd(r1-r2, m1, m2) = 1.
                # Conflict if gcd(r1-r2, m1, m2) > 1.
                # This is equivalent to: there exists a prime p dividing m1 and m2 such that r1 == r2 mod p.
                # Which is equivalent to r1 == r2 mod gcd(m1, m2).
                # So if gcd(m1, m2) = 1, then r1 == r2 mod 1 is always true.
                # So if gcd(m1, m2) = 1, they ALWAYS conflict?
                # NO.
                # Let's check: m1=2, m2=3. gcd=1.
                # r1=0, r2=0. gcd(0, 2, 3) = 1. Disjoint?
                # Classes: 0 mod 2 -> {0, 2, 4...}
                # 0 mod 3 -> {0, 3, 6...}
                # Intersection: 0, 6, 12... Not empty.
                # So they intersect.
                # The condition for DISJOINT is gcd(r1-r2, m1, m2) = 1.
                # If gcd(m1, m2) = 1, then gcd(r1-r2, 1) = 1 always.
                # So if gcd(m1, m2) = 1, they are ALWAYS disjoint?
                # Let's check the formula again.
                # System x = r1 + k1 m1, x = r2 + k2 m2.
                # r1 + k1 m1 = r2 + k2 m2
                # k1 m1 - k2 m2 = r2 - r1
                # This has a solution iff gcd(m1, m2) divides r2 - r1.
                # If gcd(m1, m2) = 1, it always has a solution.
                # So they ALWAYS intersect.
                # My previous logic was inverted.
                # Conflict (Intersection) iff gcd(m1, m2) | (r1 - r2).
                # Disjoint iff NOT (gcd(m1, m2) | (r1 - r2)).
                
                # So if g = gcd(m1, m2) == 1, then 1 | (r1-r2) is always true.
                # So they ALWAYS conflict.
                # This means we can NEVER pick two moduli that are coprime?
                # That would limit us to a set of moduli with pairwise gcd > 1.
                # Max such set?
                # 4, 6, 8, 9, 10, 12...
                # 4, 6 (gcd 2), 4, 8 (gcd 4), 6, 8 (gcd 2).
                # 4, 6, 8, 9 (gcd(4,9)=1 -> Conflict).
                # So we can't have 4 and 9.
                
                # This drastically changes the problem.
                # We need a set of moduli such that for any pair, we can pick residues that are NOT congruent mod gcd.
                # If gcd(m1, m2) = 1, we CANNOT pick them.
                # So we must pick a set of moduli where gcd(m_i, m_j) > 1 for all i!=j?
                # No, we can pick residues.
                # If gcd(m1, m2) = 1, they ALWAYS intersect. So we can't include both.
                # So the moduli we pick must form a set where gcd(m_i, m_j) > 1 for all pairs?
                # Actually, if gcd(m1, m2) = 1, they are incompatible.
                # So we need to pick a subset of moduli such that no two are coprime?
                # And then for each pair with gcd > 1, we can choose residues to be disjoint.
                # Is it always possible to choose residues for a clique of moduli (pairwise gcd > 1) such that they are pairwise disjoint?
                # Yes, likely, if we use CRT carefully or just greedy.
                
                # So the primary constraint is: Pick a subset of moduli with pairwise gcd > 1.
                # Then assign residues.
                
                # Let's re-evaluate the seed.
                # Seed moduli: 2, 4, 8, 16, 32, 64, 128.
                # Pairwise gcds are all > 1.
                # So they are compatible moduli.
                
                # Can we add 6?
                # gcd(6, 2)=2, gcd(6,4)=2, gcd(6,8)=2, etc.
                # So 6 is compatible with all seed moduli.
                # Can we pick a residue for 6 disjoint from all others?
                # Seed residues: 1 mod 2, 3 mod 4, 7 mod 8, ...
                # r_6 must satisfy:
                # r_6 != 1 mod gcd(6,2)=2 -> r_6 != 1 mod 2 -> r_6 even.
                # r_6 != 3 mod gcd(6,4)=2 -> r_6 != 1 mod 2 -> r_6 even.
                # r_6 != 7 mod gcd(6,8)=2 -> r_6 != 1 mod 2 -> r_6 even.
                # So r_6 must be even.
                # r_6 in {0, 2, 4}.
                # Check against 3 mod 4:
                # r_6 != 3 mod 2. 3 is odd. r_6 is even. OK.
                # Check against 7 mod 8:
                # r_6 != 7 mod 2. 7 is odd. r_6 is even. OK.
                # So any even r_6 works?
                # Let's check r_6 = 0.
                # 0 mod 6: {0, 6, 12, 18, 24...}
                # 1 mod 2: {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25...} (Odds)
                # Disjoint? Yes.
                # 3 mod 4: {3, 7, 11, 15, 19, 23...}
                # 0 mod 6: 0, 6, 12, 18, 24, 30...
                # Intersection with 3 mod 4?
                # 0 mod 6 is even. 3 mod 4 is odd. Disjoint.
                # So 0 mod 6 works.
                
                # So we can add 6.
                
                # The logic `if g == 1: all conflict` is correct.
                # So in the update step:
                # if g == 1:
                #    all residues in m_rem are blocked.
                # else:
                #    block r where (r - r_pick) % g == 0.
                
                if g == 1:
                    # All residues conflict
                    for r in range(m_rem):
                        blocked[idx][r] += 1
                else:
                    for r in range(m_rem):
                        if (r - r_pick) % g == 0:
                            blocked[idx][r] += 1

        return chosen_pairs

    # Run multiple times
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        current_set = greedy_random(i)
        if len(current_set) > len(best_set):
            best_set = current_set

    return [[r, m] for r, m in best_set]

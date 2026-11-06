"""Sidon set construction with conflict graph and local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Fixed elements
    fixed = [1, 2, 5, 11]
    
    # Precompute: for each candidate x in 12..284, which sums it creates with fixed elements
    # and with itself
    # A set S is Sidon on sums iff all a+b (a<=b) are distinct
    # This is equivalent to: all differences |a-b| are distinct (Golomb ruler)
    # Let's work with the sum condition directly.
    
    # Strategy: 
    # 1. Start with fixed elements
    # 2. Build a candidate pool
    # 3. Use greedy + local search with restarts
    # 4. Try to maximize size
    
    def get_sums(s):
        """Get all pairwise sums a+b where a<=b in sorted list s."""
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sums.add(s[i] + s[j])
        return sums
    
    def is_compatible(x, s, sums):
        """Check if x can be added to set s without creating duplicate sums."""
        # New sums created by adding x: x + y for y in s, and 2*x
        new_sums = set()
        for y in s:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        # Check for conflicts with existing sums
        if new_sums & sums:
            return False
        return True
    
    def greedy_fill(s, sums, candidates, rng):
        """Greedy fill: add compatible candidates in randomized order."""
        s = list(s)
        sums = set(sums)
        # Shuffle candidates for diversity
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        for x in shuffled:
            if x in s:
                continue
            if is_compatible(x, s, sums):
                s.append(x)
                for y in s:
                    sums.add(x + y)
                sums.add(2 * x)
                s.sort()
        return s, sums
    
    def local_search(s, sums, rng, max_iters=500):
        """Local search: try swapping elements to improve size."""
        s = list(s)
        sums = set(sums)
        best = len(s)
        candidates = [x for x in range(12, 285) if x not in s]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try removing one element and adding two
            for i in range(len(s)):
                if time.time() > deadline:
                    break
                removed = s[i]
                s2 = s[:i] + s[i+1:]
                # Recompute sums without removed
                sums2 = get_sums(s2)
                # Try to add two new elements
                added = 0
                for x in rng.sample(candidates, min(len(candidates), 30)):
                    if x not in s2 and is_compatible(x, s2, sums2):
                        s2.append(x)
                        for y in s2:
                            sums2.add(x + y)
                        sums2.add(2 * x)
                        s2.sort()
                        added += 1
                        if added == 2:
                            break
                if len(s2) > len(s):
                    s = s2
                    sums = sums2
                    best = len(s)
                    improved = True
                    break
            if not improved:
                # Try single swaps
                for i in range(len(s)):
                    if time.time() > deadline:
                        break
                    removed = s[i]
                    s2 = s[:i] + s[i+1:]
                    sums2 = get_sums(s2)
                    for x in rng.sample(candidates, min(len(candidates), 20)):
                        if x not in s2 and is_compatible(x, s2, sums2):
                            s2.append(x)
                            for y in s2:
                                sums2.add(x + y)
                            sums2.add(2 * x)
                            s2.sort()
                            if len(s2) > len(s):
                                s = s2
                                sums = sums2
                                best = len(s)
                                improved = True
                                break
                    if improved:
                        break
            if not improved:
                break
        return s, sums
    
    # Main search with restarts
    best_set = fixed[:]
    best_size = len(fixed)
    
    rng = random.Random(42)
    candidates = list(range(12, 285))
    
    # First, do a deterministic greedy to get a baseline
    s_init, sums_init = greedy_fill(fixed, get_sums(fixed), candidates, random.Random(0))
    best_set = s_init[:]
    best_size = len(s_init)
    
    # Now do randomized restarts with local search
    restart = 0
    while time.time() < deadline - 10:
        restart += 1
        # Randomized greedy with different seeds
        seed = rng.randint(0, 1000000)
        r = random.Random(seed)
        s, sums = greedy_fill(fixed, get_sums(fixed), candidates, r)
        
        # Local search to improve
        s, sums = local_search(s, sums, r, max_iters=200)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
        
        # If we hit 20, we're done (optimal without constraints is 20)
        if best_size >= 20:
            break
    
    # Final verification
    s_final = sorted(best_set)
    # Ensure it contains the fixed elements
    for f in fixed:
        if f not in s_final:
            s_final.append(f)
            s_final.sort()
    
    # Verify Sidon property
    sums_check = get_sums(s_final)
    expected = len(s_final) * (len(s_final) + 1) // 2
    if len(sums_check) != expected:
        # Fallback to valid set
        s_final = fixed + [x for x in range(12, 285) if x not in fixed][:best_size - 4]
        # Re-verify
        sums_check = get_sums(s_final)
        while len(sums_check) != len(s_final) * (len(s_final) + 1) // 2:
            s_final = s_final[:-1]
            sums_check = get_sums(s_final)
    
    return s_final

# EVOLVE-BLOCK-END

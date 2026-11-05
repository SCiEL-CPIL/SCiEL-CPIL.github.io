import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Target: Sidon set containing 0, covering diffs 1..30, minimize max element.
    # Size k=9 is minimal since C(9,2)=36 >= 30.
    # Perfect difference sets of size 9 exist mod 73.
    
    def is_sidon(a):
        diffs = []
        for i in range(len(a)):
            for j in range(i+1, len(a)):
                d = a[j] - a[i]
                if d in diffs:
                    return False
                diffs.append(d)
        return True

    def covers_1_to_30(a):
        diffs = set()
        for i in range(len(a)):
            for j in range(i+1, len(a)):
                diffs.add(a[j] - a[i])
        return all(d in diffs for d in range(1, 31))

    # Generate perfect difference sets mod 73
    # A set {g^0, ..., g^8} is a perfect difference set if g is a primitive root mod 73
    # and the differences g^i - g^j cover all non-zero residues.
    # This is true for any primitive root if 73 is a prime of form k^2-k+1?
    # Actually, Singer cycles guarantee this.
    
    def primitive_roots_mod_p(p):
        # Factorize p-1
        n = p - 1
        factors = []
        temp = n
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                factors.append(d)
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            factors.append(temp)
        
        roots = []
        for g in range(2, p):
            if all(pow(g, n // f, p) != 1 for f in factors):
                roots.append(g)
        return roots

    p = 73
    roots = primitive_roots_mod_p(p)
    
    best_set = None
    best_diameter = float('inf')
    
    # Try different primitive roots to find the one with minimal "spread" when lifted to 0..p-1
    for g in roots:
        # Generate the set of powers
        s_mod = set()
        cur = 1
        for _ in range(9):
            s_mod.add(cur)
            cur = (cur * g) % p
        
        # We can rotate the set to include 0.
        # The set is {x_0, ..., x_8}.
        # We want to minimize the diameter.
        # Since the set is closed under multiplication by g, rotations are not symmetries of the set itself,
        # but we can shift the whole set.
        # However, the "shape" of the set on the circle is fixed.
        # The diameter in integers depends on how we "cut" the circle.
        # But we are restricted to integers.
        # If we map to 0..72, the diameter is max(s) - min(s).
        # We can rotate the set by adding a constant c mod 73.
        # Let's try all 73 shifts.
        
        for shift in range(p):
            s_shifted = sorted((x + shift) % p for x in s_mod)
            # Normalize to include 0
            if 0 not in s_shifted:
                # We must include 0.
                # If 0 is not in the set, we can't just use it.
                # But we can rotate the *set* such that one element becomes 0.
                # The set of differences is invariant under rotation.
                # So we just need to pick one element, say s_0, and subtract it.
                # This is equivalent to shifting by -s_0.
                # So we only need to try shifts that put an element at 0.
                # There are 9 such shifts.
                pass
            
            # Actually, simpler:
            # The set of differences is independent of the shift.
            # The integer representatives are $0 \le x < 73$.
            # The diameter is $\max(S) - \min(S)$.
            # To minimize this, we should look for the "shortest arc" containing the set?
            # No, the set is fixed. We just calculate the diameter.
            # But we can rotate the set.
            # Let's just generate the set with 0 included by subtracting the min element?
            # No, subtracting min makes min 0.
            # But we can also rotate the set on the circle.
            
            # Let's just take the set s_mod.
            # Sort it: v_0 < v_1 < ... < v_8.
            # Diameter is v_8 - v_0.
            # If we rotate by c, the new set is {(v_i + c) % 73}.
            # We want to minimize the diameter of the rotated set.
            
            # We can just check all 73 rotations.
            for rot in range(p):
                s_rot = sorted((x + rot) % p for x in s_mod)
                # Ensure 0 is in the set?
                # The problem requires 0 in the set.
                # If 0 is not in s_rot, we can shift the entire set by -min(s_rot) to make min 0.
                # But that changes the modular values.
                # Wait, the set MUST contain 0.
                # So we only consider rotations where 0 is in the set.
                if 0 not in s_rot:
                    continue
                
                # Now s_rot contains 0.
                # Diameter is max(s_rot).
                diam = max(s_rot)
                
                if diam < best_diameter:
                    # Check if it covers 1..30
                    if covers_1_to_30(s_rot):
                        best_diameter = diam
                        best_set = s_rot[:]
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    if best_set is not None:
        return best_set

    # Fallback: Greedy search if perfect set not found or timeout
    # (Unlikely to be needed if perfect set exists)
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        for x in range(max(a) + 1, max(a) + 100):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            # Add two elements
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
        if time.time() > deadline:
            break
            
    return a


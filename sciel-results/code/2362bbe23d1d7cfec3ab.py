import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def paley_orientation(p):
        """Return orientation list for Paley tournament on p vertices."""
        squares = set()
        for x in range(1, p):
            squares.add(x * x % p)
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    def cyclic_orientation(n):
        """Return orientation list for standard Cyclic tournament on n vertices (n odd).
        i -> j if (j - i) % n <= (n-1)//2
        """
        orientation = []
        half = (n - 1) // 2
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff <= half:
                    orientation.append(0) # i -> j
                else:
                    orientation.append(1) # j -> i
        return orientation

    def build_in_neigh_masks_python(n, orientation):
        """Build in-neighbor bitmask (Python int) for each vertex."""
        in_neigh = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                if bit == 0:
                    # i -> j, so i is in-neighbor of j
                    in_neigh[j] |= (1 << i)
                else:
                    # j -> i, so j is in-neighbor of i
                    in_neigh[i] |= (1 << j)
                idx += 1
        return in_neigh

    def check_tournament_fast(n, orientation):
        """Check if every 4-set has a common dominator using Python int bitmasks."""
        if time.time() - start_time > DEADLINE:
            return False
        
        in_neigh = build_in_neigh_masks_python(n, orientation)
        
        # Check all 4-sets using bit operations
        for a in range(n):
            if time.time() - start_time > DEADLINE:
                return False
            for b in range(a + 1, n):
                ab = in_neigh[a] & in_neigh[b]
                if ab == 0:
                    return False
                for c in range(b + 1, n):
                    abc = ab & in_neigh[c]
                    if abc == 0:
                        return False
                    for d in range(c + 1, n):
                        if (abc & in_neigh[d]) == 0:
                            return False
        return True

    def count_bad_4sets_python(n, orientation):
        """Count number of 4-sets without a common dominator."""
        in_neigh = build_in_neigh_masks_python(n, orientation)
        bad_count = 0
        for a in range(n):
            for b in range(a + 1, n):
                ab = in_neigh[a] & in_neigh[b]
                if ab == 0:
                    # All 4-sets containing a,b where c,d > b are bad
                    # c ranges from b+1 to n-2, d from c+1 to n-1
                    for c in range(b + 1, n - 1):
                        bad_count += (n - 1 - c)
                    continue
                for c in range(b + 1, n):
                    abc = ab & in_neigh[c]
                    if abc == 0:
                        for d in range(c + 1, n):
                            bad_count += 1
                        continue
                    for d in range(c + 1, n):
                        if (abc & in_neigh[d]) == 0:
                            bad_count += 1
        return bad_count

    def random_tournament(n):
        """Generate a random tournament orientation."""
        orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        return orientation

    def flip_edge(orientation, n, i, j):
        """Flip edge between i and j (i < j) in orientation list."""
        idx = i * (2 * n - i - 1) // 2 + (j - i - 1)
        orientation[idx] = 1 - orientation[idx]

    def repair_tournament(n, orientation, max_time=10.0):
        """Local repair: flip edges to minimize bad 4-sets."""
        current_bad = count_bad_4sets_python(n, orientation)
        if current_bad == 0:
            return orientation, True
        
        deadline = start_time + max_time
        no_improve_count = 0
        max_no_improve = 200 # Reduced to avoid long stalls
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            in_neigh = build_in_neigh_masks_python(n, orientation)
            
            bad_found = False
            best_flip = None
            best_new_bad = current_bad + 1
            
            # Sample some bad 4-sets
            attempts = 0
            max_attempts = 200
            while attempts < max_attempts and not bad_found:
                attempts += 1
                a = random.randint(0, n - 4)
                b = random.randint(a + 1, n - 3)
                c = random.randint(b + 1, n - 2)
                d = random.randint(c + 1, n - 1)
                
                abc = in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
                if abc == 0:
                    bad_found = True
                    verts = [a, b, c, d]
                    # Try flipping each edge in the 4-set
                    for i in range(4):
                        for j in range(i + 1, 4):
                            u, v = verts[i], verts[j]
                            if u > v:
                                u, v = v, u
                            
                            flip_edge(orientation, n, u, v)
                            new_bad = count_bad_4sets_python(n, orientation)
                            if new_bad < best_new_bad:
                                best_new_bad = new_bad
                                best_flip = (u, v)
                            
                            # Revert
                            flip_edge(orientation, n, u, v)
                    break
            
            if not bad_found:
                break
            
            if best_flip is not None and best_new_bad < current_bad:
                u, v = best_flip
                flip_edge(orientation, n, u, v)
                current_bad = best_new_bad
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        final_bad = count_bad_4sets_python(n, orientation)
        return orientation, final_bad == 0

    # Strategy:
    # 1. Try Cyclic Tournaments for odd n (distinct family from Paley)
    # 2. Try Paley Tournaments for primes p=3 mod 4
    # 3. Try Random + Repair for smaller n
    # 4. Fallback to known valid n=67
    
    best_result = None
    best_n = 91
    
    # Phase 1: Try Cyclic Tournaments
    for n in range(31, 70, 2):
        if time.time() - start_time > DEADLINE:
            break
        orientation = cyclic_orientation(n)
        if check_tournament_fast(n, orientation):
            best_result = {'n': n, 'orientation': orientation}
            best_n = n
            break
    
    if best_result is not None:
        return best_result

    # Phase 2: Try Paley Tournaments
    for p in range(31, 70):
        if time.time() - start_time > DEADLINE:
            break
        if p % 4 != 3 or not is_prime(p):
            continue
        orientation = paley_orientation(p)
        if check_tournament_fast(p, orientation):
            best_result = {'n': p, 'orientation': orientation}
            best_n = p
            break
    
    if best_result is not None:
        return best_result

    # Phase 3: Try Random + Repair for smaller n (31 to 45)
    upper_n = 45
    for n in range(31, upper_n + 1):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Try a few random restarts
        max_restarts = 2
        found = False
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            random.seed(42 + restart * 1000 + n)
            orientation = random_tournament(n)
            repaired, is_valid = repair_tournament(n, orientation, max_time=5.0)
            
            if is_valid:
                best_result = {'n': n, 'orientation': repaired}
                best_n = n
                found = True
                break
        
        if found:
            break
    
    if best_result is not None:
        return best_result

    # Phase 4: Fallback to Paley p=67 (known valid)
    p = 67
    orientation = paley_orientation(p)
    # We trust 67 is valid based on problem statement, but verify if time permits
    if time.time() - start_time < DEADLINE - 2:
        if check_tournament_fast(p, orientation):
            return {'n': p, 'orientation': orientation}
    
    # If verification failed (shouldn't happen) or time is out, return it anyway as best guess
    return {'n': p, 'orientation': orientation}

import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def get_paley_67():
        p = 67
        squares = {x*x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                if ((j - i) % p) in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': p, 'orientation': orientation}

    def count_bad_fast(orient, n):
        in_m = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if orient[idx] == 0:
                    in_m[j] |= (1 << i)
                else:
                    in_m[i] |= (1 << j)
                idx += 1
        
        bad = 0
        for a in range(n):
            ia = in_m[a]
            for b in range(a+1, n):
                iab = ia & in_m[b]
                if iab == 0:
                    rem = n - b - 1
                    if rem >= 2:
                        bad += rem * (rem - 1) // 2
                    continue
                for c in range(b+1, n):
                    iabc = iab & in_m[c]
                    if iabc == 0:
                        bad += n - c - 1
                        continue
                    for d in range(c+1, n):
                        if (iabc & in_m[d]) == 0:
                            bad += 1
        return bad

    def orient_to_in_m(orient, n):
        in_m = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if orient[idx] == 0:
                    in_m[j] |= (1 << i)
                else:
                    in_m[i] |= (1 << j)
                idx += 1
        return in_m

    def count_bad_from_in_m(in_m, n):
        bad = 0
        for a in range(n):
            ia = in_m[a]
            for b in range(a+1, n):
                iab = ia & in_m[b]
                if iab == 0:
                    rem = n - b - 1
                    if rem >= 2:
                        bad += rem * (rem - 1) // 2
                    continue
                for c in range(b+1, n):
                    iabc = iab & in_m[c]
                    if iabc == 0:
                        bad += n - c - 1
                        continue
                    for d in range(c+1, n):
                        if (iabc & in_m[d]) == 0:
                            bad += 1
        return bad

    # Try to find a valid tournament with n=66 by removing a vertex from Paley-67
    if time.time() < deadline - 5:
        p = 67
        squares = {x*x % p for x in range(1, p)}
        full_orient = []
        for i in range(p):
            for j in range(i+1, p):
                if ((j - i) % p) in squares:
                    full_orient.append(0)
                else:
                    full_orient.append(1)
        
        # Try removing each vertex and check if the resulting 66-vertex tournament is valid
        for remove_vertex in range(p):
            if time.time() > deadline - 3:
                break
            
            # Build the orientation for the 66-vertex tournament (excluding remove_vertex)
            n = 66
            remap = []
            for v in range(p):
                if v != remove_vertex:
                    remap.append(v)
            
            # Build new orientation
            new_orient = []
            idx = 0
            for i in range(p):
                for j in range(i+1, p):
                    if i == remove_vertex or j == remove_vertex:
                        idx += 1
                        continue
                    # Find positions in remap
                    if i < remove_vertex:
                        ni = i
                    else:
                        ni = i - 1
                    if j < remove_vertex:
                        nj = j
                    else:
                        nj = j - 1
                    # The bit in full_orient at index idx
                    new_orient.append(full_orient[idx])
                    idx += 1
            
            # Verify it's valid
            in_m = orient_to_in_m(new_orient, n)
            bad = count_bad_from_in_m(in_m, n)
            
            if bad == 0:
                return {'n': n, 'orientation': new_orient}
    
    # Try local search starting from a random 66-vertex tournament
    if time.time() < deadline - 10:
        random.seed(42)
        n = 66
        n_pairs = n * (n - 1) // 2
        orientation = [random.choice([0, 1]) for _ in range(n_pairs)]
        in_m = orient_to_in_m(orientation, n)
        cur_bad = count_bad_from_in_m(in_m, n)
        
        if cur_bad == 0:
            return {'n': n, 'orientation': orientation}
            
        # Local search with incremental updates
        for step in range(500):
            if time.time() > deadline - 5:
                break
            idx = random.randint(0, n_pairs - 1)
            i = 0
            count = 0
            while count + (n - 1 - i) > idx:
                i += 1
                count += (n - 1 - i)
            j = i + 1 + (idx - count)
            
            old_val = orientation[idx]
            orientation[idx] = 1 - old_val
            
            # Update in_m incrementally
            if old_val == 0:
                # Was i->j, now j->i
                in_m[j] &= ~(1 << i)
                in_m[i] |= (1 << j)
            else:
                # Was j->i, now i->j
                in_m[i] &= ~(1 << j)
                in_m[j] |= (1 << i)
            
            new_bad = count_bad_from_in_m(in_m, n)
            
            if new_bad < cur_bad:
                cur_bad = new_bad
                if cur_bad == 0:
                    return {'n': n, 'orientation': orientation}
            else:
                # Revert
                orientation[idx] = old_val
                if old_val == 0:
                    in_m[j] |= (1 << i)
                    in_m[i] &= ~(1 << j)
                else:
                    in_m[i] |= (1 << j)
                    in_m[j] &= ~(1 << i)
    
    return get_paley_67()

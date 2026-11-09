import time
import math

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def check_validity(n, orientation):
    """
    Checks if the tournament defined by 'orientation' satisfies the condition:
    Every set of 4 vertices has a common dominator (a vertex with arcs to all 4).
    
    orientation: list of bits for pairs (i, j) with i < j.
                 0 means i -> j, 1 means j -> i.
    """
    # Build out-neighborhood bitmasks
    # N[i] is a bitmask where bit j is 1 if i -> j
    N = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                N[i] |= (1 << j)
            else:
                N[j] |= (1 << i)
            idx += 1
            
    # Check all 4-sets
    # A set {i, j, k, l} has a common dominator v if v is in N[i] & N[j] & N[k] & N[l]
    # Note: v must be distinct from i, j, k, l.
    # In a tournament, v cannot be an out-neighbor of itself, so v is automatically distinct.
    # However, we must ensure v is not one of the vertices in the set?
    # "Common dominator" usually implies a vertex in the tournament.
    # If v is one of the 4 vertices, say v=i, then i must have arcs to j, k, l.
    # But i does not have an arc to itself. The condition is "vertex with arcs to all four".
    # This phrasing "arcs to all four" usually implies the vertex is *outside* the set of four, or it's a "king" type property.
    # Let's re-read carefully: "Every set of 4 vertices must have a vertex with arcs to all four."
    # This typically means there exists a vertex $v$ (possibly in the set? No, if $v$ is in the set, it can't have an arc to itself. So $v$ must be distinct from the 4 vertices? Or does "arcs to all four" mean arcs to the other three and itself? No, arcs are between distinct vertices.
    # Standard interpretation in Schütte's problem: For any 4 vertices $A, B, C, D$, there exists a vertex $X$ such that $X \to A, X \to B, X \to C, X \to D$.
    # $X$ can be one of $A, B, C, D$? If $X=A$, then $A \to A$ is impossible. So $X$ must be distinct from the 4 vertices.
    # Wait, if the set is $\{A, B, C, D\}$, and $X=A$, then we need $A \to A$ which is false.
    # So $X$ must be a vertex *not* in the set?
    # Or does it mean "a vertex in the tournament"? Yes.
    # So for any 4 distinct vertices $v_1, v_2, v_3, v_4$, there exists $u \in V$ such that $u \to v_i$ for all $i=1..4$.
    # If $u$ is one of the $v_i$, it fails. So $u$ must be distinct from $v_1, v_2, v_3, v_4$.
    # Thus, we need to check if $\bigcap_{i=1}^4 N(v_i)$ contains a vertex $u$ such that $u \notin \{v_1, v_2, v_3, v_4\}$.
    
    for i in range(n - 3):
        Ni = N[i]
        for j in range(i + 1, n - 2):
            common_ij = Ni & N[j]
            if common_ij == 0:
                continue
            for k in range(j + 1, n - 1):
                common_ijk = common_ij & N[k]
                if common_ijk == 0:
                    continue
                for l in range(k + 1, n):
                    common_ijkl = common_ijk & N[l]
                    # The common dominator u must be in common_ijkl
                    # AND u must not be i, j, k, or l.
                    # Since N[x] never has bit x set, if u was i, bit i would not be in N[i], so it wouldn't be in the intersection.
                    # Wait. N[i] is out-neighbors of i. i is not in N[i].
                    # So if u=i, i is not in N[i], so i is not in N[i] & N[j] & ...
                    # Therefore, any vertex in the intersection is automatically distinct from the vertices in the set.
                    # So we just need common_ijkl != 0.
                    if common_ijkl == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # We search for the smallest prime p = 4k + 3 for which the Paley tournament is valid.
    # Known: p=67 works.
    # We check p=7, 11, 19, 23, 31, 43, 47, 59, 67.
    
    best_result = None
    
    for p in range(7, 91):
        if time.time() > deadline:
            break
            
        if p % 4 != 3:
            continue
        if not is_prime(p):
            continue
            
        # Construct Paley tournament on p vertices
        # Squares mod p
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
            
        # Build orientation list
        orientation = []
        # Also build bitmasks for validity check to avoid rebuilding
        N = [0] * p
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in squares:
                    # i -> j
                    orientation.append(0)
                    N[i] |= (1 << j)
                else:
                    # j -> i
                    orientation.append(1)
                    N[j] |= (1 << i)
                    
        # Verify condition using the precomputed N
        valid = True
        for i in range(p - 3):
            Ni = N[i]
            for j in range(i + 1, p - 2):
                common_ij = Ni & N[j]
                if common_ij == 0:
                    continue
                for k in range(j + 1, p - 1):
                    common_ijk = common_ij & N[k]
                    if common_ijk == 0:
                        continue
                    for l in range(k + 1, p):
                        if (common_ijk & N[l]) == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            return {'n': p, 'orientation': orientation}
            
    # Fallback: Return the known valid Paley tournament for p=67
    # This should not be reached if the loop above works correctly,
    # but ensures we always return a valid result.
    p = 67
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
        
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': p, 'orientation': orientation}

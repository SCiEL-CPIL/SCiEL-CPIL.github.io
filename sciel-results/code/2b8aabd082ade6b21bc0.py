"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def count_violations(adj, n):
        """Count 4-sets without a common dominator (dominator outside the set)."""
        violations = 0
        for a in range(n - 3):
            for b in range(a + 1, n - 2):
                for c in range(b + 1, n - 1):
                    for d in range(c + 1, n):
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                        found = False
                        for v in range(n):
                            if v == a or v == b or v == c or v == d:
                                continue
                            if (adj[v] & mask) == mask:
                                found = True
                                break
                        if not found:
                            violations += 1
        return violations

    def try_local_search(adj, n, max_iter=500):
        """Try to fix violations by flipping edges."""
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
            v = count_violations(adj, n)
            if v == 0:
                return adj, n, True
            # Find a violating 4-set and flip an edge to help
            for a in range(n - 3):
                for b in range(a + 1, n - 2):
                    for c in range(b + 1, n - 1):
                        for d in range(c + 1, n):
                            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            found = False
                            for v2 in range(n):
                                if v2 == a or v2 == b or v2 == c or v2 == d:
                                    continue
                                if (adj[v2] & mask) == mask:
                                    found = True
                                    break
                            if not found:
                                # This 4-set is bad. Try flipping an edge to create a dominator.
                                # For each candidate dominator v2 outside the set, 
                                # check how many of the 4 it already dominates.
                                best_flip = None
                                best_gain = 0
                                for v2 in range(n):
                                    if v2 == a or v2 == b or v2 == c or v2 == d:
                                        continue
                                    current = sum(1 for x in (a,b,c,d) if (adj[v2] >> x) & 1)
                                    if current < 4:
                                        # Need to flip edges from v2 to the missing ones
                                        missing = [x for x in (a,b,c,d) if not ((adj[v2] >> x) & 1)]
                                        # Flipping one edge: v2 -> x becomes x -> v2
                                        # This helps if v2 was already dominating 3 of 4
                                        if len(missing) == 1:
                                            x = missing[0]
                                            # Flip edge between v2 and x
                                            if (adj[v2] >> x) & 1:
                                                # v2 -> x, need to flip to x -> v2
                                                adj[v2] &= ~(1 << x)
                                                adj[x] |= (1 << v2)
                                                best_flip = (v2, x, True)  # True means we did the flip
                                                break
                                            else:
                                                # x -> v2, need to flip to v2 -> x
                                                adj[x] &= ~(1 << v2)
                                                adj[v2] |= (1 << x)
                                                best_flip = (v2, x, False)
                                                break
                                if best_flip:
                                    break
                            if best_flip:
                                break
                        if best_flip:
                            break
                    if best_flip:
                        break
                if best_flip:
                    break
        v = count_violations(adj, n)
        return adj, n, v == 0

    # Start from Paley tournament on 83 vertices
    p = 83
    squares = {x * x % p for x in range(1, p)}
    n = p
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % p in squares:
                adj[i] |= (1 << j)
            else:
                adj[j] |= (1 << i)

    # Try to reduce n by removing vertices
    for _ in range(30):
        if time.time() > deadline - 30:
            break
        removed = False
        for v in range(n - 1, -1, -1):
            if time.time() > deadline - 30:
                break
            new_n = n - 1
            new_adj = []
            for i in range(n):
                if i == v:
                    continue
                new_bits = 0
                for j in range(n):
                    if i == j or j == v:
                        continue
                    if (adj[i] >> j) & 1:
                        new_j = j if j < v else j - 1
                        new_bits |= (1 << new_j)
                new_adj.append(new_bits)
            
            # Check validity with sampling
            import random as _random
            valid = True
            for _ in range(500):
                if time.time() > deadline - 30:
                    break
                idxs = _random.sample(range(new_n), 4)
                a, b, c, d = idxs
                mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                found = False
                for vv in range(new_n):
                    if vv in (a, b, c, d):
                        continue
                    if (new_adj[vv] & mask) == mask:
                        found = True
                        break
                if not found:
                    valid = False
                    break
            if valid:
                n = new_n
                adj = new_adj
                removed = True
                break
        if not removed:
            break

    # Try local search to fix any remaining issues
    adj, n, valid = try_local_search(adj, n, max_iter=200)

    # Build orientation
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                orientation.append(0)
            else:
                orientation.append(1)

    return {'n': n, 'orientation': orientation}
# EVOLVE-BLOCK-END

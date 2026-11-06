"""Snake-in-the-box Q_9: Warnsdorff backtracking with bitmask state + greedy refinement."""
import time
import random
import sys

N = 9
MAX_V = 1 << N
MAX_TIME = 115.0  # Safety margin before 120s evaluator limit

# Precompute neighbors for all vertices
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(MAX_V)]

# Precompute neighbor bitmask for each vertex (bit v is set if v is a neighbor)
# This allows O(1) check: is vertex u a neighbor of vertex v? -> (mask[v] >> u) & 1
_NEIGHBOR_MASK = [0] * MAX_V
for v in range(MAX_V):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    _NEIGHBOR_MASK[v] = mask

# Precompute bit count for fast popcount if needed
_BIT_COUNT = [i.bit_count() for i in range(MAX_V)]


def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME

    def time_left():
        return deadline - time.time()

    def verify_snake(path):
        """Verify that path is a valid induced path in Q_N."""
        if not path:
            return False
        used = set(path)
        if len(used) != len(path):
            return False
        for i in range(len(path)):
            for j in range(i + 1, len(path)):
                diff = path[i] ^ path[j]
                if abs(i - j) == 1:
                    if diff == 0 or (diff & (diff - 1)) != 0:
                        return False
                else:
                    if diff == 0 or (diff & (diff - 1)) == 0:
                        return False
        return True

    def greedy_extend(path, used, forbidden, bit_order, max_steps=512):
        """Extend path greedily using forbidden set."""
        for _ in range(max_steps):
            if time_left() < 1.0:
                break
            cur = path[-1]
            found = False
            for b in bit_order:
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    for nb in _NEIGHBORS[cur]:
                        forbidden.add(nb)
                    path.append(v)
                    used.add(v)
                    found = True
                    break
            if not found:
                break
        return path

    def greedy_with_lookahead(path, used, forbidden, bit_order, max_steps=512):
        """Greedy with lookahead scoring."""
        for _ in range(max_steps):
            if time_left() < 1.0:
                break
            cur = path[-1]
            best_candidate = None
            best_score = -1

            candidates = []
            for b in bit_order:
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    candidates.append(v)

            if not candidates:
                break

            for v in candidates:
                score = 0
                for b2 in bit_order:
                    v2 = v ^ (1 << b2)
                    if v2 == cur:
                        continue
                    if v2 not in used and v2 not in forbidden:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = v

            if best_candidate is None:
                break

            for nb in _NEIGHBORS[cur]:
                forbidden.add(nb)
            path.append(best_candidate)
            used.add(best_candidate)

        return path

    def warnsdorff_backtrack(start_vertex, deadline, randomize_ties=False):
        """
        Deep backtracking search using Warnsdorff's heuristic.
        State: (current, forbidden_mask, path)
        forbidden_mask: bit v is 1 if v is used OR v is a neighbor of a non-terminal vertex.
        When moving from cur to v:
          new_forbidden = forbidden | _NEIGHBOR_MASK[cur] | (1 << v)
        Valid next vertex u: (forbidden >> u) & 1 == 0 AND u is neighbor of current.
        """
        sys.setrecursionlimit(100000)

        best_path = [start_vertex]
        best_len = 1

        # Initial state
        path = [start_vertex]
        forbidden = 1 << start_vertex  # start vertex is used
        current = start_vertex

        def backtrack(current, forbidden, path):
            nonlocal best_path, best_len

            if time.time() > deadline:
                return False

            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]

            # Find valid neighbors of current
            valid_neighbors = []
            for v in _NEIGHBORS[current]:
                if (forbidden >> v) & 1:
                    continue

                # Calculate Warnsdorff score: number of valid neighbors of v
                # if we move to v, new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
                new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)

                count = 0
                for u in _NEIGHBORS[v]:
                    if not ((new_forbidden >> u) & 1):
                        count += 1
                valid_neighbors.append((count, v))

            if not valid_neighbors:
                return False

            # Sort by Warnsdorff score (ascending - fewest options first)
            # Add randomness for tie-breaking to explore different branches
            if randomize_ties:
                random.shuffle(valid_neighbors)
            valid_neighbors.sort(key=lambda x: x[0])

            for count, v in valid_neighbors:
                if time.time() > deadline:
                    return False

                new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
                path.append(v)

                if backtrack(v, new_forbidden, path):
                    path.pop()
                    return True

                path.pop()

            return False

        backtrack(current, forbidden, path)
        return best_path

    def local_improvement(path, bit_order, max_rounds=10):
        """Try to extend path from both ends greedily."""
        best_path = list(path)

        for _ in range(max_rounds):
            if time_left() < 2.0:
                break
            improved = False

            # Try extending forward
            used = set(best_path)
            forbidden = set()
            for i in range(len(best_path) - 1):
                for nb in _NEIGHBORS[best_path[i]]:
                    forbidden.add(nb)

            cur = best_path[-1]
            for b in bit_order:
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    best_path.append(v)
                    improved = True
                    break

            # Try extending backward
            if not improved:
                used = set(best_path)
                forbidden = set()
                for i in range(1, len(best_path)):
                    for nb in _NEIGHBORS[best_path[i]]:
                        forbidden.add(nb)

                cur = best_path[0]
                for b in bit_order:
                    v = cur ^ (1 << b)
                    if v not in used and v not in forbidden:
                        best_path.insert(0, v)
                        improved = True
                        break

            if not improved:
                break

        return best_path

    best_path = [0]
    best_len = 1

    # Phase 1: Run Warnsdorff backtracking from vertex 0
    # This is the primary search method - deep, structured exploration
    p = warnsdorff_backtrack(0, deadline, randomize_ties=False)
    if len(p) > best_len:
        best_path = p
        best_len = len(p)

    # Phase 2: If time remains, try more backtracking runs with different parameters
    if time_left() > 5.0 and best_len < 200:
        # Try with randomized tie-breaking
        p = warnsdorff_backtrack(0, deadline, randomize_ties=True)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)

    # Phase 3: Local improvement on best path
    if time_left() > 3.0:
        bo = list(range(N))
        random.shuffle(bo)
        improved = local_improvement(best_path, bo, max_rounds=20)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)

    # Phase 4: If still time, try greedy from random starts to find alternative paths
    if time_left() > 10.0:
        for _ in range(50):
            if time_left() < 5.0:
                break
            sv = random.randint(0, MAX_V - 1)
            bo = list(range(N))
            random.shuffle(bo)
            path = [sv]
            used = {sv}
            forbidden = set()
            p = greedy_with_lookahead(path, used, forbidden, bo, max_steps=300)
            # Try to extend this greedy result with backtracking
            if len(p) > 50:  # Only worth trying if greedy got reasonably far
                # Use the greedy path as initial state for a short backtracking
                # Actually, let's just do local improvement
                bo2 = list(range(N))
                random.shuffle(bo2)
                imp = local_improvement(p, bo2, max_rounds=5)
                if len(imp) > best_len:
                    best_path = imp
                    best_len = len(imp)

    # Phase 5: Final local improvement
    if time_left() > 2.0:
        bo = list(range(N))
        random.shuffle(bo)
        improved = local_improvement(best_path, bo, max_rounds=20)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)

    return best_path

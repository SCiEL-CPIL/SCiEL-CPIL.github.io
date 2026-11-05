"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 9
TIME_LIMIT = 145  # safety margin below 150s

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_path = None
    best_len = 0

    def elapsed():
        return time.time() - start_time

    def time_left():
        return TIME_LIMIT - elapsed()

    def is_valid_extension(path, used, candidate):
        """Check if candidate can be appended to path (induced path constraint)."""
        if candidate in used:
            return False
        # Must differ from all non-consecutive vertices by at least 2 bits
        # Consecutive is path[-1], which we already know differs by 1
        for i in range(len(path) - 2):
            if (candidate ^ path[i]).bit_count() == 1:
                return False
        return True

    def greedy_extend(path, used, bit_order, max_steps=None):
        """Extend path greedily using given bit order. Returns extended path."""
        if max_steps is None:
            max_steps = 512
        for _ in range(max_steps):
            cur = path[-1]
            found = False
            for b in bit_order:
                v = cur ^ (1 << b)
                if is_valid_extension(path, used, v):
                    path.append(v)
                    used.add(v)
                    found = True
                    break
            if not found:
                break
        return path

    def greedy_with_backtracking(path, used, bit_order, max_depth=30):
        """Extend with limited backtracking. Returns extended path."""
        if max_depth <= 0:
            return greedy_extend(path, used, bit_order)

        cur = path[-1]
        candidates = []
        for b in bit_order:
            v = cur ^ (1 << b)
            if is_valid_extension(path, used, v):
                candidates.append((b, v))

        if not candidates:
            return path

        # Try each candidate, recurse with reduced depth
        best_result = list(path)
        for b, v in candidates:
            if time_left() < 1.0:
                break
            path.append(v)
            used.add(v)
            result = greedy_with_backtracking(path, used, bit_order, max_depth - 1)
            if len(result) > len(best_result):
                best_result = result
            path.pop()
            used.discard(v)

        return best_result

    def try_greedy_with_lookahead(start_vertex, bit_order, max_steps=512):
        """Greedy with simple lookahead: at each step, try each valid bit and pick the one
        that allows the longest further greedy extension (limited depth)."""
        path = [start_vertex]
        used = {start_vertex}

        for _ in range(max_steps):
            cur = path[-1]
            best_candidate = None
            best_score = -1

            for b in bit_order:
                v = cur ^ (1 << b)
                if not is_valid_extension(path, used, v):
                    continue
                # Score: prefer bits that leave more options (simple heuristic)
                # Count how many further extensions are possible from v
                score = 0
                for b2 in bit_order:
                    v2 = v ^ (1 << b2)
                    if v2 == cur:
                        continue
                    # Quick check: v2 not in used, and not adjacent to path[:-1]
                    if v2 not in used:
                        valid = True
                        for i in range(len(path) - 2):
                            if (v2 ^ path[i]).bit_count() == 1:
                                valid = False
                                break
                        if valid:
                            score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = v

            if best_candidate is None:
                break

            path.append(best_candidate)
            used.add(best_candidate)

        return path

    def random_bit_order():
        order = list(range(N))
        random.shuffle(order)
        return order

    # First: try the original greedy (lowest bit first) to get baseline
    path0 = greedy_extend([0], {0}, list(range(N)))
    best_path = path0
    best_len = len(path0)

    # Try different starting vertices and bit orders
    start_vertices = [0]
    # Add some random starts
    for _ in range(20):
        start_vertices.append(random.randint(0, 2**N - 1))

    attempts = 0
    for sv in start_vertices:
        if time_left() < 2.0:
            break
        for _ in range(5):  # 5 random bit orders per start
            if time_left() < 2.0:
                break
            bo = random_bit_order()
            path = try_greedy_with_lookahead(sv, bo)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path
            attempts += 1

    # Try backtracking from the best path found
    if time_left() > 5.0 and best_path:
        used = set(best_path)
        # Try extending with backtracking from current end
        extended = greedy_with_backtracking(list(best_path), used, random_bit_order(), max_depth=20)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended

    # More random restarts with greedy+lookahead
    while time_left() > 3.0:
        sv = random.randint(0, 2**N - 1)
        bo = random_bit_order()
        path = try_greedy_with_lookahead(sv, bo)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    return best_path
# EVOLVE-BLOCK-END


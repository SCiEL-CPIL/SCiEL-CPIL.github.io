"""Snake-in-the-box Q_9: Warnsdorff backtracking + greedy + local improvement."""
import time
import random
import sys

N = 9
TIME_LIMIT = 115  # safety margin below 120s hard kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def time_left():
        return TIME_LIMIT - elapsed()

    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c

    def get_neighbors(v):
        """Generate all N neighbors of v."""
        for b in range(N):
            yield v ^ (1 << b)

    def is_valid_snake(path):
        """Verify path is a valid induced path in Q_N."""
        if not path:
            return False
        used = set(path)
        if len(used) != len(path):
            return False
        n = len(path)
        for i in range(n):
            for j in range(i + 1, n):
                diff = path[i] ^ path[j]
                if diff == 0:
                    return False
                pc = popcount(diff)
                if abs(i - j) == 1:
                    if pc != 1:
                        return False
                else:
                    if pc == 1:
                        return False
        return True

    def warnsdorff_backtrack(start_vertex, bit_order, max_steps=512, max_backtracks=200):
        """
        Warnsdorff's rule with limited backtracking.
        At each step, choose the neighbor with the fewest valid continuations.
        If stuck, backtrack and try the next-best option.
        """
        path = [start_vertex]
        used = {start_vertex}
        # forbidden: vertices that would create a distance-1 conflict with a non-adjacent path vertex
        # When we add vertex v at position i, all neighbors of path[i-1] (except v itself) become forbidden
        # because they'd be distance 1 from path[i-1] but not adjacent to v in the path
        forbidden = set()

        # We'll use a stack for backtracking
        # Each state: (path_copy, used_copy, forbidden_copy, next_index_in_bit_order)
        # To save memory, we'll do limited backtracking with a depth cap

        best_path = list(path)
        best_len = 1
        backtracks_used = 0
        max_backtracks_local = max_backtracks

        # Use iterative deepening approach with Warnsdorff
        # Actually, let's do a simple recursive-like approach with a stack

        # Simpler: greedy Warnsdorff with occasional random tie-breaking
        # and restart if stuck early
        for step in range(max_steps):
            if time_left() < 2.0:
                break

            cur = path[-1]
            candidates = []
            for b in bit_order:
                v = cur ^ (1 << b)
                if v in used or v in forbidden:
                    continue
                # Count valid continuations from v
                count = 0
                for b2 in bit_order:
                    v2 = v ^ (1 << b2)
                    if v2 == cur:
                        continue
                    if v2 not in used and v2 not in forbidden:
                        count += 1
                candidates.append((count, random.random(), v))

            if not candidates:
                # Dead end - try backtracking one step
                if len(path) > 2:
                    # Remove last vertex and try a different branch
                    # Simple backtracking: remove last, mark its neighbors as forbidden differently
                    # For simplicity, just break and let outer loop restart
                    break
                break

            # Sort by (count ascending, random for tie-breaking)
            candidates.sort(key=lambda x: (x[0], x[1]))
            chosen = candidates[0][2]

            # Commit
            for nb in get_neighbors(cur):
                if nb != chosen:
                    forbidden.add(nb)
            path.append(chosen)
            used.add(chosen)

            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)

        return best_path

    def warnsdorff_with_backtracking(start_vertex, bit_order, max_steps=512, max_depth=15):
        """
        Warnsdorff with actual backtracking up to max_depth levels.
        Uses a stack of states.
        """
        path = [start_vertex]
        used = {start_vertex}
        forbidden = set()
        best_path = list(path)
        best_len = 1

        # Stack entries: (path, used, forbidden, candidate_index)
        # We limit backtracking depth to avoid explosion
        backtrack_stack = []
        max_bt = max_depth

        for step in range(max_steps):
            if time_left() < 2.0:
                break

            cur = path[-1]
            candidates = []
            for b in bit_order:
                v = cur ^ (1 << b)
                if v in used or v in forbidden:
                    continue
                count = 0
                for b2 in bit_order:
                    v2 = v ^ (1 << b2)
                    if v2 == cur:
                        continue
                    if v2 not in used and v2 not in forbidden:
                        count += 1
                candidates.append((count, random.random(), v))

            if not candidates:
                # Dead end
                if backtrack_stack and len(backtrack_stack) <= max_bt:
                    # Pop last state
                    saved_path, saved_used, saved_forbidden, next_idx = backtrack_stack.pop()
                    path = saved_path
                    used = saved_used
                    forbidden = saved_forbidden
                    # Try the next candidate from the saved position
                    # We need to re-derive candidates... this is complex
                    # Simplified: just break
                    break
                else:
                    break

            # Sort by Warnsdorff
            candidates.sort(key=lambda x: (x[0], x[1]))

            # Save state for potential backtracking (only save top choices)
            if len(backtrack_stack) < max_bt and len(candidates) > 1:
                # Save current state before choosing
                # Save the second-best option
                backtrack_stack.append((
                    list(path),
                    set(used),
                    set(forbidden),
                    1  # index of next candidate to try
                ))

            chosen = candidates[0][2]

            # Commit
            for nb in get_neighbors(cur):
                if nb != chosen:
                    forbidden.add(nb)
            path.append(chosen)
            used.add(chosen)

            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)

        return best_path

    def greedy_extend(path, used, forbidden, bit_order, max_steps=512):
        """Simple greedy extension."""
        for _ in range(max_steps):
            cur = path[-1]
            found = False
            for b in bit_order:
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    for nb in get_neighbors(cur):
                        forbidden.add(nb)
                    path.append(v)
                    used.add(v)
                    found = True
                    break
            if not found:
                break
        return path

    def greedy_with_lookahead(path, used, forbidden, bit_order, max_steps=512):
        """Greedy with 2-step lookahead."""
        for _ in range(max_steps):
            cur = path[-1]
            best_candidate = None
            best_score = -1

            for b in bit_order:
                v = cur ^ (1 << b)
                if v in used or v in forbidden:
                    continue
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

            for nb in get_neighbors(cur):
                forbidden.add(nb)
            path.append(best_candidate)
            used.add(best_candidate)

        return path

    def local_improvement(path, bit_order, max_rounds=5):
        """Try to extend path from both ends."""
        best_path = list(path)

        for _ in range(max_rounds):
            if time_left() < 3.0:
                break

            improved = False
            for direction in [1, -1]:
                cur = best_path[-1] if direction == 1 else best_path[0]
                used = set(best_path)

                if direction == 1:
                    forbidden = set()
                    for i in range(len(best_path) - 1):
                        for nb in get_neighbors(best_path[i]):
                            forbidden.add(nb)
                else:
                    forbidden = set()
                    for i in range(1, len(best_path)):
                        for nb in get_neighbors(best_path[i]):
                            forbidden.add(nb)

                for b in bit_order:
                    v = cur ^ (1 << b)
                    if v not in used and v not in forbidden:
                        if direction == 1:
                            best_path.append(v)
                        else:
                            best_path.insert(0, v)
                        improved = True
                        break

            if not improved:
                break

        return best_path

    def random_bit_order():
        order = list(range(N))
        random.shuffle(order)
        return order

    best_path = None
    best_len = 0

    # Phase 1: Warnsdorff attempts (primary strategy)
    phase1_deadline = start_time + min(50.0, time_left() * 0.5)

    # Deterministic first attempt
    p = warnsdorff_backtrack(0, list(range(N)), max_steps=512)
    best_path = p
    best_len = len(p)

    # Many random Warnsdorff restarts
    attempts = 0
    while time.time() < phase1_deadline and time_left() > 5.0:
        sv = random.randint(0, 2**N - 1)
        bo = random_bit_order()
        p = warnsdorff_backtrack(sv, bo, max_steps=512)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        attempts += 1

    # Phase 2: Warnsdorff with backtracking (deeper search, fewer attempts)
    phase2_deadline = start_time + min(time_left() * 0.3, 30.0)

    while time.time() < phase2_deadline and time_left() > 5.0:
        sv = random.randint(0, 2**N - 1)
        bo = random_bit_order()
        p = warnsdorff_with_backtracking(sv, bo, max_steps=512, max_depth=10)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    # Phase 3: Local improvement on best path
    if time_left() > 5.0:
        bo = random_bit_order()
        improved = local_improvement(best_path, bo, max_rounds=10)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)

    # Phase 4: More Warnsdorff attempts with time remaining
    if time_left() > 8.0:
        while time_left() > 3.0:
            sv = random.randint(0, 2**N - 1)
            bo = random_bit_order()
            p = warnsdorff_backtrack(sv, bo, max_steps=512)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
            # Quick local improvement
            if len(p) > best_len - 5:
                imp = local_improvement(p, bo, max_rounds=3)
                if len(imp) > best_len:
                    best_len = len(imp)
                    best_path = imp

    # Phase 5: Greedy fallback attempts
    if time_left() > 5.0:
        for _ in range(30):
            if time_left() < 2.0:
                break
            sv = random.randint(0, 2**N - 1)
            bo = random_bit_order()
            path = [sv]
            used = {sv}
            forbidden = set()
            p = greedy_with_lookahead(path, used, forbidden, bo, max_steps=512)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p

    # Final local improvement
    if time_left() > 2.0 and best_path is not None:
        bo = random_bit_order()
        improved = local_improvement(best_path, bo, max_rounds=5)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)

    return best_path
# EVOLVE-BLOCK-END

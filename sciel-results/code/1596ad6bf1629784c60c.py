import time
import random

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 115.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    best_path = []
    best_len = 0

    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))

    def count_onward_moves(u, used, adj_count):
        """Count valid onward moves if u were added to path."""
        cnt = 0
        for w in neighbors[u]:
            if not used[w] and adj_count[w] == 0:
                cnt += 1
        return cnt

    def try_extend(path, used, adj_count, is_left):
        """Try to extend path at one end. Returns (success, new_vertex)."""
        end_v = path[0] if is_left else path[-1]
        valid = []
        for u in neighbors[end_v]:
            if not used[u] and adj_count[u] == 0:
                cnt = count_onward_moves(u, used, adj_count)
                valid.append((cnt, u))
        if not valid:
            return False, None
        # Sort by onward count (Warnsdorff-like), pick from top tier
        valid.sort()
        best_score = valid[0][0]
        candidates = [x for x in valid if x[0] <= best_score + 1]
        _, next_v = random.choice(candidates)
        return True, next_v

    def bidirectional_grow(path, time_limit):
        """Grow a snake bidirectionally. Returns the path."""
        nonlocal best_path, best_len
        if len(path) < 2:
            return path

        used = [False] * MAX_VERTS
        adj_count = [0] * MAX_VERTS
        for v in path:
            used[v] = True
            for w in neighbors[v]:
                adj_count[w] += 1

        local_best = path[:]

        def update_best(p):
            nonlocal best_path, best_len, local_best
            if len(p) > len(local_best):
                local_best = p[:]
            if len(p) > best_len:
                best_len = len(p)
                best_path = p[:]

        update_best(path)

        while time.time() - start_time < time_limit:
            # Try left
            ok_left, left_v = try_extend(path, used, adj_count, True)
            # Try right
            ok_right, right_v = try_extend(path, used, adj_count, False)

            if not ok_left and not ok_right:
                break

            if ok_left and not ok_right:
                path.insert(0, left_v)
                used[left_v] = True
                for w in neighbors[left_v]:
                    adj_count[w] += 1
            elif ok_right and not ok_left:
                path.append(right_v)
                used[right_v] = True
                for w in neighbors[right_v]:
                    adj_count[w] += 1
            else:
                # Both valid: pick end with fewer options (Warnsdorff)
                # Re-count to compare
                left_opts = 0
                for u in neighbors[path[0]]:
                    if not used[u] and adj_count[u] == 0:
                        left_opts += 1
                right_opts = 0
                for u in neighbors[path[-1]]:
                    if not used[u] and adj_count[u] == 0:
                        right_opts += 1

                if left_opts <= right_opts:
                    path.insert(0, left_v)
                    used[left_v] = True
                    for w in neighbors[left_v]:
                        adj_count[w] += 1
                else:
                    path.append(right_v)
                    used[right_v] = True
                    for w in neighbors[right_v]:
                        adj_count[w] += 1

            update_best(path)

        return local_best

    def quick_greedy(v_start, max_steps=50):
        """Generate a short greedy path from v_start."""
        path = [v_start]
        used = [False] * MAX_VERTS
        adj_count = [0] * MAX_VERTS
        used[v_start] = True
        for w in neighbors[v_start]:
            adj_count[w] += 1

        for _ in range(max_steps):
            v_end = path[-1]
            valid = []
            for u in neighbors[v_end]:
                if not used[u] and adj_count[u] == 0:
                    cnt = count_onward_moves(u, used, adj_count)
                    valid.append((cnt, u))
            if not valid:
                break
            valid.sort()
            best_score = valid[0][0]
            candidates = [x for x in valid if x[0] <= best_score + 1]
            _, next_v = random.choice(candidates)
            path.append(next_v)
            used[next_v] = True
            for w in neighbors[next_v]:
                adj_count[w] += 1

        return path

    def local_repair(path, time_limit):
        """Truncate and regrow to improve a snake."""
        nonlocal best_path, best_len
        if not path or len(path) < 3:
            return path

        current = path[:]

        for k in [1, 2, 3, 4, 5, 6, 8]:
            if time.time() - start_time > time_limit:
                break
            if len(current) <= k + 2:
                break

            truncated = current[:-k]
            if len(truncated) < 2:
                continue

            # Rebuild state
            used = [False] * MAX_VERTS
            adj_count = [0] * MAX_VERTS
            for v in truncated:
                used[v] = True
                for w in neighbors[v]:
                    adj_count[w] += 1

            new_path = truncated[:]
            # Regrow right
            for _ in range(k + 10):  # Try to recover and exceed
                if time.time() - start_time > time_limit:
                    break
                v_end = new_path[-1]
                valid = []
                for u in neighbors[v_end]:
                    if not used[u] and adj_count[u] == 0:
                        cnt = count_onward_moves(u, used, adj_count)
                        valid.append((cnt, u))
                if not valid:
                    break
                valid.sort()
                best_score = valid[0][0]
                candidates = [x for x in valid if x[0] <= best_score + 1]
                _, next_v = random.choice(candidates)
                new_path.append(next_v)
                used[next_v] = True
                for w in neighbors[next_v]:
                    adj_count[w] += 1

            if len(new_path) > len(current):
                current = new_path[:]
                if len(current) > best_len:
                    best_len = len(current)
                    best_path = current[:]

        return current

    # Phase 1: Quick greedy baselines
    for _ in range(20):
        if time.time() - start_time > DEADLINE - 10:
            break
        v0 = random.randint(0, MAX_VERTS - 1)
        p = quick_greedy(v0, max_steps=30)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]

    # Phase 2: Bidirectional growth with multiple starts
    attempts = 0
    while time.time() - start_time < DEADLINE - 5:
        attempts += 1
        remaining = DEADLINE - (time.time() - start_time)
        per_attempt = min(3.0, remaining / 10.0)
        if per_attempt < 0.05:
            break

        # Start with a short greedy path
        v0 = random.randint(0, MAX_VERTS - 1)
        initial = quick_greedy(v0, max_steps=15)
        if len(initial) < 2:
            continue

        p = bidirectional_grow(initial, start_time + per_attempt)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]

    # Phase 3: Local repair on best path
    if best_path and len(best_path) > 5:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 2:
            repaired = local_repair(best_path, start_time + remaining - 1)
            if len(repaired) > best_len:
                best_len = len(repaired)
                best_path = repaired[:]

    if not best_path:
        best_path = [0, 1]

    return best_path

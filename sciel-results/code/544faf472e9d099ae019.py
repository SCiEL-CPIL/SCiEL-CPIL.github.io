"""Difference triangle set (4,7): bitset-accelerated search."""
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 140.0

    def greedy_construct():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    d = x - a
                    if d > 0 and (used >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        d = x - a
                        used |= (1 << d)
            rows.append(row)
        return rows

    greedy_rows = greedy_construct()
    best_scope = max(max(row) for row in greedy_rows)
    best_rows = greedy_rows

    def row_to_bits(row):
        bits = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                bits |= (1 << d)
        return bits

    def generate_rows(scope, used_bits, max_rows=2000):
        """Yield rows of k+1 entries starting at 0, max entry <= scope,
        with no differences conflicting with used_bits."""
        count = 0
        def backtrack(pos, current, last_val):
            nonlocal count
            if time.time() > deadline or count >= max_rows:
                return
            if pos == k + 1:
                bits = row_to_bits(current)
                if not (bits & used_bits):
                    count += 1
                    yield current[:]
                return
            for v in range(last_val + 1, scope + 1):
                ok = True
                for a in current:
                    d = v - a
                    if (used_bits >> d) & 1:
                        ok = False
                        break
                if ok:
                    current.append(v)
                    yield from backtrack(pos + 1, current, v)
                    current.pop()
        yield from backtrack(1, [0], 0)

    # Iterative deepening
    for target_scope in range(best_scope - 1, 0, -1):
        if time.time() > deadline - 15.0:
            break

        candidates = list(generate_rows(target_scope, 0, max_rows=3000))
        if not candidates:
            continue

        def search_rows(idx, used_bits, chosen):
            if time.time() > deadline:
                return None
            if idx == n_rows:
                return chosen[:]
            for row in candidates:
                bits = row_to_bits(row)
                if bits & used_bits:
                    continue
                result = search_rows(idx + 1, used_bits | bits, chosen + [row])
                if result is not None:
                    return result
            return None

        result = search_rows(0, 0, [])
        if result is not None:
            all_diffs = set()
            valid = True
            for row in result:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
            if valid:
                scope = max(max(row) for row in result)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = result
                break

    return best_rows
# EVOLVE-BLOCK-END

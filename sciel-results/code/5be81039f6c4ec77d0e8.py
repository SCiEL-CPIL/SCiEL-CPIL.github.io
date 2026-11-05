def construct():
    import time
    start_time = time.time()
    deadline = start_time + 80

    from math import gcd
    def lcm(a, b):
        return a * b // gcd(a, b)

    N = 128
    lcm_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Find all bad triples
    bad_triples = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = lcm_table[a][b]
            for c in range(b+1, N+1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))

    # Count how many bad triples each element is in
    triple_count = [0]*(N+1)
    for a, b, c in bad_triples:
        triple_count[a] += 1
        triple_count[b] += 1
        triple_count[c] += 1

    # Delete approach: start with all, remove elements in bad triples
    selected = set(range(1, N+1))

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for a, b, c in bad_triples:
            if a in selected and b in selected and c in selected:
                max_elem = max([a, b, c], key=lambda x: triple_count[x])
                selected.remove(max_elem)
                changed = True
                break

    excluded = set(range(1, N+1)) - selected

    def can_add(x, current_set):
        for a in current_set:
            for b in current_set:
                if a < b:
                    if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                        return False
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in sorted(excluded):
            if time.time() > deadline:
                break
            if can_add(x, selected):
                selected.add(x)
                excluded.remove(x)
                improved = True

    return sorted(selected)

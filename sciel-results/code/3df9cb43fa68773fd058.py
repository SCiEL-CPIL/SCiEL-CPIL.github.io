import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    N = 180
    all_elements = set(range(1, N + 1))
    
    # Precompute: for each a, all pairs (b,c) with b<c, b>a, c>a, (b-a)(c-a)=a², b,c<=N
    a_pairs = {a: [] for a in range(1, N + 1)}
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, a + 1):
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                if b <= N and c <= N and b != c:
                    a_pairs[a].append((b, c))
    
    # Triple count for prioritization
    triple_count = {x: 0 for x in range(1, N + 1)}
    for a in range(1, N + 1):
        for (b, c) in a_pairs[a]:
            triple_count[a] += 1
            triple_count[b] += 1
            triple_count[c] += 1
    
    def is_valid(S):
        S_set = set(S)
        for a in S:
            for (b, c) in a_pairs[a]:
                if b in S_set and c in S_set:
                    return False
        return True
    
    # Start with all, remove to make valid
    S = set(range(1, N + 1))
    changed = True
    while changed and (time.time() - start_time) < DEADLINE:
        changed = False
        S_set = set(S)
        for a in sorted(S):
            for (b, c) in a_pairs[a]:
                if b in S_set and c in S_set:
                    candidates = [a, b, c]
                    to_remove = max(candidates, key=lambda x: triple_count[x])
                    S.discard(to_remove)
                    S_set.discard(to_remove)
                    changed = True
                    break
            if changed:
                break
    
    # Add back removed elements
    removed = all_elements - S
    for x in sorted(removed, key=lambda y: -triple_count[y]):
        if (time.time() - start_time) > DEADLINE:
            break
        S_test = S | {x}
        if is_valid(S_test):
            S = S_test
    
    return sorted(S)

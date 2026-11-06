import time
import itertools

N = 40

def construct():
    start = time.time()
    deadline = start + 80.0  # Safety margin before 90s limit

    # All possible undirected chords (excluding rim edges)
    # A chord is (u,v) with 0 <= u < v < N and not (v == u+1 or (u==0 and v==N-1))
    all_chords = []
    for u in range(N):
        for v in range(u + 1, N):
            # Skip rim edges
            if v == u + 1:
                continue
            if u == 0 and v == N - 1:
                continue
            all_chords.append((u, v))

    # For each chord, compute the cycle lengths it covers
    # Chord (u,v): shorter arc length = (v-u) edges -> cycle length v-u+1
    #              longer arc length = (N-(v-u)) edges -> cycle length N-(v-u)+1
    chord_lengths = {}
    for (u, v) in all_chords:
        d = v - u
        l1 = d + 1
        l2 = N - d + 1
        lens = set()
        if 3 <= l1 <= N:
            lens.add(l1)
        if 3 <= l2 <= N:
            lens.add(l2)
        chord_lengths[(u, v)] = lens

    # We need to cover lengths 3..N-1 (length N is the rim)
    target_lengths = set(range(3, N))

    # Greedy set cover
    def greedy_cover():
        uncovered = set(target_lengths)
        chosen = []
        remaining = list(all_chords)
        while uncovered and remaining:
            best_chord = None
            best_count = 0
            for chord in remaining:
                cnt = len(chord_lengths[chord] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_chord = chord
            if best_chord is None:
                break
            chosen.append(best_chord)
            uncovered -= chord_lengths[best_chord]
            remaining.remove(best_chord)
        return chosen, uncovered

    # Local search: try to improve by swapping
    def local_search(chords, max_iters=200):
        chords = list(chords)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            uncovered = target_lengths - set()
            for c in chords:
                uncovered -= chord_lengths[c]
            if not uncovered:
                return chords
            # Try swapping one chord for another
            improved = False
            for i in range(len(chords)):
                for j in range(len(chords)):
                    if i == j:
                        continue
                    # Try replacing chords[i] with some other chord
                    for new_chord in all_chords:
                        if new_chord in chords:
                            continue
                        new_set = chords[:i] + [new_chord] + chords[i+1:]
                        new_uncovered = target_lengths - set()
                        for c in new_set:
                            new_uncovered -= chord_lengths[c]
                        if not new_uncovered:
                            chords = new_set
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return chords

    # Build the solution from a set of chords
    def build_solution(chords):
        chords = list(set(chords))
        edges = []
        # Rim edges
        for i in range(N):
            edges.append((i, (i + 1) % N))
        # Chord edges
        for (u, v) in chords:
            edges.append((u, v))

        # Map each length to a cycle
        # Length N: rim
        length_map = {N: ('rim',)}

        for (u, v) in chords:
            d = v - u
            l1 = d + 1
            l2 = N - d + 1
            if l1 not in length_map and 3 <= l1 <= N:
                # Cycle: u, u+1, ..., v (length d+1)
                length_map[l1] = ('short', u, v)
            if l2 not in length_map and 3 <= l2 <= N:
                # Cycle: u, u-1, u-2, ..., v (going the long way)
                length_map[l2] = ('long', u, v)

        # Check all lengths covered
        for L in range(3, N + 1):
            if L not in length_map:
                return None

        # Generate cycles
        cycles = []
        for L in range(3, N + 1):
            entry = length_map[L]
            if entry[0] == 'rim':
                cycles.append(list(range(N)))
            elif entry[0] == 'short':
                u, v = entry[1], entry[2]
                # u, u+1, ..., v
                cyc = list(range(u, v + 1))
                cycles.append(cyc)
            elif entry[0] == 'long':
                u, v = entry[1], entry[2]
                # u, u-1, u-2, ... wrapping around to v
                cyc = [u]
                cur = (u - 1) % N
                while cur != v:
                    cyc.append(cur)
                    cur = (cur - 1) % N
                cyc.append(v)
                cycles.append(cyc)

        # Verify: all cycle vertices distinct and edges exist
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((min(a, b), max(a, b)))

        for cyc in cycles:
            if len(cyc) != len(set(cyc)):
                return None
            for i in range(len(cyc)):
                a = cyc[i]
                b = cyc[(i + 1) % len(cyc)]
                if (min(a, b), max(a, b)) not in edge_set:
                    return None

        return {'edges': edges, 'cycles': cycles}

    # Try greedy
    chords, uncovered = greedy_cover()
    if not uncovered:
        result = build_solution(chords)
        if result:
            return result

    # Try local search from greedy
    chords2 = local_search(chords)
    result = build_solution(chords2)
    if result:
        return result

    # Fallback: try star chords from vertex 0 (original approach, deterministic)
    # Chord (0,k) covers lengths k+1 and N-k+1
    star_chords = []
    uncovered = set(target_lengths)
    for k in range(2, N - 1):
        l1 = k + 1
        l2 = N - k + 1
        if l1 in uncovered or l2 in uncovered:
            star_chords.append((0, k))
            uncovered.discard(l1)
            uncovered.discard(l2)
        if not uncovered:
            break
    if not uncovered:
        result = build_solution(star_chords)
        if result:
            return result

    # Final fallback: fan graph (always valid)
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = []
    for L in range(3, N + 1):
        cycles.append(list(range(L)))
    return {'edges': edges, 'cycles': cycles}

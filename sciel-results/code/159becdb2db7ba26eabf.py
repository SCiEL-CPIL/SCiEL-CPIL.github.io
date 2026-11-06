import time
import itertools
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def get_lengths_from_chords(chords):
        lengths = {N}
        for (a, b) in chords:
            d = b - a
            l1 = d + 1
            l2 = N - d + 1
            if 3 <= l1 <= N:
                lengths.add(l1)
            if 3 <= l2 <= N:
                lengths.add(l2)
        
        for i in range(len(chords)):
            for j in range(i+1, len(chords)):
                a1, b1 = chords[i]
                a2, b2 = chords[j]
                if a1 < a2 < b2 < b1:
                    L = (a2 - a1) + (b1 - b2) + 2
                    if 3 <= L <= N:
                        lengths.add(L)
                elif a2 < a1 < b1 < b2:
                    L = (a1 - a2) + (b2 - b1) + 2
                    if 3 <= L <= N:
                        lengths.add(L)
        return lengths

    def is_pancyclic(chords):
        lengths = get_lengths_from_chords(chords)
        return all(L in lengths for L in range(3, N + 1))

    def build_result(chords):
        edges = set()
        for i in range(N):
            edges.add((i, (i + 1) % N))
        for (a, b) in chords:
            edges.add((a, b))
        
        edge_list = list(edges)
        cycles = []
        length_map = {}
        
        for (a, b) in chords:
            d = b - a
            l1 = d + 1
            l2 = N - d + 1
            if 3 <= l1 <= N and l1 not in length_map:
                length_map[l1] = ('short', a, b)
            if 3 <= l2 <= N and l2 not in length_map:
                length_map[l2] = ('long', a, b)
        
        for i in range(len(chords)):
            for j in range(i+1, len(chords)):
                a1, b1 = chords[i]
                a2, b2 = chords[j]
                if a1 < a2 < b2 < b1:
                    L = (a2 - a1) + (b1 - b2) + 2
                    if 3 <= L <= N and L not in length_map:
                        length_map[L] = ('nested', a1, a2, b2, b1)
                elif a2 < a1 < b1 < b2:
                    L = (a1 - a2) + (b2 - b1) + 2
                    if 3 <= L <= N and L not in length_map:
                        length_map[L] = ('nested', a2, a1, b1, b2)
        
        length_map[N] = ('rim',)
        
        for L in range(3, N + 1):
            if L not in length_map:
                return None
            typ = length_map[L]
            if typ[0] == 'rim':
                cycles.append(list(range(N)))
            elif typ[0] == 'short':
                a, b = typ[1], typ[2]
                cycles.append(list(range(a, b + 1)))
            elif typ[0] == 'long':
                a, b = typ[1], typ[2]
                cycles.append([a] + list(range(b, N)) + list(range(0, a)))
            elif typ[0] == 'nested':
                a1, a2, b2, b1 = typ[1], typ[2], typ[3], typ[4]
                cycles.append(list(range(a1, a2 + 1)) + list(range(b2, b1 + 1)))
        
        return {'edges': edge_list, 'cycles': cycles}

    def greedy_find_chords(m, seed=0):
        rng = random.Random(seed)
        chords = []
        all_candidates = [(a, b) for a in range(N) for b in range(a+2, N) if b-a <= N-2]
        
        for _ in range(m):
            best_chord = None
            best_gain = -1
            current_lengths = get_lengths_from_chords(chords)
            current_covered = current_lengths.intersection(set(range(3, N)))
            
            for chord in all_candidates:
                if chord in chords:
                    continue
                new_chords = chords + [chord]
                new_lengths = get_lengths_from_chords(new_chords)
                new_covered = new_lengths.intersection(set(range(3, N)))
                gain = len(new_covered) - len(current_covered)
                if gain > best_gain:
                    best_gain = gain
                    best_chord = chord
            
            if best_chord is None or best_gain <= 0:
                break
            chords.append(best_chord)
            if is_pancyclic(chords):
                break
        
        return chords if is_pancyclic(chords) else None

    # Try minimal chord counts
    for m in [5, 6, 7, 8]:
        if time.time() > deadline:
            break
        for attempt in range(30):
            if time.time() > deadline:
                break
            chords = greedy_find_chords(m, seed=attempt)
            if chords:
                result = build_result(chords)
                if result:
                    return result
        # Random restarts
        for attempt in range(100):
            if time.time() > deadline:
                break
            rng = random.Random(10000 + m * 100 + attempt)
            all_candidates = [(a, b) for a in range(N) for b in range(a+2, N) if b-a <= N-2]
            chords = rng.sample(all_candidates, m)
            if is_pancyclic(chords):
                result = build_result(chords)
                if result:
                    return result

    # Fallback: fan graph
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = [list(range(L)) for L in range(3, N + 1)]
    return {'edges': edges, 'cycles': cycles}

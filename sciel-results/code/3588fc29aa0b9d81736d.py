import time
import random

N = 40

def _validate(edges, cycles):
    n = N
    edge_set = set()
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))

    if len(cycles) != n - 2:
        return False

    for i, cycle in enumerate(cycles):
        L = i + 3
        if len(cycle) != L:
            return False
        if len(set(cycle)) != L:
            return False
        for j in range(L):
            u = cycle[j]
            v = cycle[(j + 1) % L]
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                return False
    return True

def _find_cycle_on_hamiltonian(chord_set, L):
    """Find a cycle of length L in a graph with Hamiltonian cycle 0-1-...-39-0 plus chords."""
    if L == N:
        return list(range(N))
    
    # A cycle of length L can be formed by taking a contiguous arc of the Hamiltonian cycle
    # and replacing one segment with a chord, or by combining multiple chords.
    
    # Strategy: Try all pairs of chords that could form a cycle of length L.
    # A cycle using chord (a,b) and chord (c,d) exists if the arcs between them work out.
    
    # Simpler: For each pair of distinct chords, check if they can form a cycle of length L.
    # Actually, let's use a BFS/DFS approach on the small graph.
    
    # Build adjacency for the subgraph we care about
    adj = {}
    for i in range(N):
        adj[i] = set()
    
    # Add Hamiltonian cycle edges
    for i in range(N):
        j = (i + 1) % N
        adj[i].add(j)
        adj[j].add(i)
    
    # Add chords
    for a, b in chord_set:
        adj[a].add(b)
        adj[b].add(a)
    
    # Now find a cycle of length L using DFS
    # This is expensive, so let's be smarter.
    
    # Key insight: In a Hamiltonian cycle + chords graph, a cycle of length L
    # can be formed by taking a path of length L-1 along the Hamiltonian cycle
    # and closing it with a chord, OR by combining two chords.
    
    # Case 1: One chord closes a path
    # If we have chord (a, b), then for any L, we need a path of length L-1 from a to b
    # along the Hamiltonian cycle. The distance along the Hamiltonian cycle from a to b
    # is min((b-a) % N, (a-b) % N). If this distance is L-1 or N-(L-1), we're done.
    
    for a, b in chord_set:
        d1 = (b - a) % N
        d2 = (a - b) % N
        if d1 == L - 1 or d2 == L - 1:
            # Construct the cycle
            if d1 == L - 1:
                cycle = [a] + [(a + i) % N for i in range(1, L)]
                return cycle
            else:
                cycle = [a] + [(a - i) % N for i in range(1, L)]
                return cycle
    
    # Case 2: Two chords
    # With chords (a,b) and (c,d), we can form cycles by combining arcs.
    # This is more complex. Let's try a different approach.
    
    # Actually, let's just do a proper search for a cycle of length L.
    # Use DFS with pruning.
    
    def dfs_cycle(start, current, visited, path, remaining):
        if remaining == 0:
            # Check if we can close the cycle
            if start in adj[current]:
                return path + [start]
            return None
        if len(visited) >= L:
            return None
        
        for neighbor in sorted(adj[current]):
            if neighbor == start and remaining > 0:
                continue
            if neighbor in visited:
                continue
            visited.add(neighbor)
            path.append(neighbor)
            result = dfs_cycle(start, neighbor, visited, path, remaining - 1)
            if result:
                return result
            path.pop()
            visited.remove(neighbor)
        return None
    
    # Try starting from each vertex
    for start in range(N):
        visited = {start}
        path = [start]
        result = dfs_cycle(start, start, visited, path, L - 1)
        if result:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Start with the known 59-edge solution: Hamiltonian cycle + 19 chords from 0
    # Then try to reduce the number of chords while maintaining pancyclicity.
    
    # The 59-edge solution has chords (0, d) for d in 2..20
    # Let's search for a smaller set of chords.
    
    # Key insight: We need cycles of every length 3 to 40.
    # The Hamiltonian cycle gives us length 40.
    # Chords create shorter cycles.
    
    # Let's use a greedy approach to select chords that cover as many cycle lengths as possible.
    
    def get_covered_lengths(chord_set):
        """Return the set of cycle lengths (3..39) that can be formed."""
        covered = set()
        
        # Build adjacency
        adj = {}
        for i in range(N):
            adj[i] = set()
        for i in range(N):
            j = (i + 1) % N
            adj[i].add(j)
            adj[j].add(i)
        for a, b in chord_set:
            adj[a].add(b)
            adj[b].add(a)
        
        for L in range(3, N):
            if L in covered:
                continue
            
            # Try to find a cycle of length L
            found = False
            
            # Case 1: Single chord
            for a, b in chord_set:
                d1 = (b - a) % N
                d2 = (a - b) % N
                if d1 == L - 1 or d2 == L - 1:
                    covered.add(L)
                    found = True
                    break
            if found:
                continue
            
            # Case 2: Two chords - this is complex, so we do a quick search
            # For efficiency, only do this for small L or when necessary
            if L <= 15:
                # DFS for cycle of length L
                def dfs(start, current, visited, path, remaining):
                    if remaining == 0:
                        if start in adj[current]:
                            return True
                        return False
                    for neighbor in adj[current]:
                        if neighbor == start and remaining > 0:
                            continue
                        if neighbor in visited:
                            continue
                        visited.add(neighbor)
                        path.append(neighbor)
                        if dfs(start, neighbor, visited, path, remaining - 1):
                            return True
                        path.pop()
                        visited.remove(neighbor)
                    return False
                
                for start in range(N):
                    visited = {start}
                    path = [start]
                    if dfs(start, start, visited, path, L - 1):
                        covered.add(L)
                        found = True
                        break
                if found:
                    continue
            
            # If we couldn't find it quickly, assume it's not covered for now
            # (This may lead to overestimation of needed chords, but that's safe)
        
        return covered
    
    # Greedy chord selection
    all_possible_chords = []
    for i in range(N):
        for j in range(i + 1, N):
            if j == (i + 1) % N or i == (j + 1) % N:
                continue  # Skip Hamiltonian cycle edges
            all_possible_chords.append((i, j))
    
    random.shuffle(all_possible_chords)
    
    best_chords = set()
    best_covered = set()
    
    # Try multiple random orderings
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        random.shuffle(all_possible_chords)
        chords = set()
        covered = set()
        
        for chord in all_possible_chords:
            if len(chords) >= 19:
                break
            new_chords = chords | {chord}
            new_covered = get_covered_lengths(new_chords)
            if len(new_covered) > len(covered):
                chords = new_chords
                covered = new_covered
            if len(covered) == 37:  # All lengths 3..39
                break
        
        if len(chords) < len(best_chords) or (len(chords) == len(best_chords) and len(covered) > len(best_covered)):
            best_chords = set(chords)
            best_covered = set(covered)
        
        if len(best_chords) <= 15:
            break
    
    # If we found a good set, use it; otherwise fall back to the known solution
    if len(best_chords) <= 19 and len(best_covered) >= 37:
        chord_list = list(best_chords)
    else:
        # Fall back to the known 19-chord solution
        chord_list = [(0, d) for d in range(2, 21)]
    
    # Build edges
    ham_edges = [(i, (i + 1) % N) for i in range(N)]
    edges = ham_edges + chord_list
    
    # Build cycles
    cycles = []
    adj = {}
    for i in range(N):
        adj[i] = set()
    for i in range(N):
        j = (i + 1) % N
        adj[i].add(j)
        adj[j].add(i)
    for a, b in chord_list:
        adj[a].add(b)
        adj[b].add(a)
    
    for L in range(3, N + 1):
        if L == N:
            cycles.append(list(range(N)))
            continue
        
        # Find cycle of length L
        cycle = None
        
        # Try single chord first
        for a, b in chord_list:
            d1 = (b - a) % N
            d2 = (a - b) % N
            if d1 == L - 1:
                cycle = [a] + [(a + i) % N for i in range(1, L)]
                break
            elif d2 == L - 1:
                cycle = [a] + [(a - i) % N for i in range(1, L)]
                break
        if cycle:
            cycles.append(cycle)
            continue
        
        # Try DFS
        def dfs(start, current, visited, path, remaining):
            if remaining == 0:
                if start in adj[current]:
                    return path + [start]
                return None
            for neighbor in sorted(adj[current]):
                if neighbor == start and remaining > 0:
                    continue
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                path.append(neighbor)
                result = dfs(start, neighbor, visited, path, remaining - 1)
                if result:
                    return result
                path.pop()
                visited.remove(neighbor)
            return None
        
        for start in range(N):
            visited = {start}
            path = [start]
            result = dfs(start, start, visited, path, L - 1)
            if result:
                cycle = result
                break
        
        if cycle:
            cycles.append(cycle)
        else:
            # This shouldn't happen if our chord set is valid
            # Fallback: use a simple cycle
            cycles.append(list(range(L)))
    
    if not _validate(edges, cycles):
        # Fallback to 59-edge solution
        ham_edges = [(i, (i + 1) % N) for i in range(N)]
        chord_edges = [(0, d) for d in range(2, 21)]
        fallback_edges = ham_edges + chord_edges
        fallback_cycles = []
        for Ll in range(3, N + 1):
            if Ll == N:
                fallback_cycles.append(list(range(N)))
            elif Ll <= 21:
                fallback_cycles.append(list(range(Ll)))
            else:
                end_vertex = N + 1 - Ll
                fallback_cycles.append([0] + list(range(N - 1, end_vertex - 1, -1)))
        return {'edges': fallback_edges, 'cycles': fallback_cycles}
    
    return {'edges': edges, 'cycles': cycles}

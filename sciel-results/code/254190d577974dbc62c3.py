import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Identify largest connected component to focus search
    visited_comp = [False] * n
    largest_comp = []
    for i in range(n):
        if not visited_comp[i]:
            stack = [i]
            comp = []
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            if len(comp) > len(largest_comp):
                largest_comp = comp

    if not largest_comp:
        return [0]

    best_path = []
    best_len = 0

    # Helper: check if a path is valid
    def is_valid_path(path):
        if not path:
            return False
        if len(set(path)) != len(path):
            return False
        for i in range(len(path) - 1):
            if path[i + 1] not in adj[path[i]]:
                return False
        return True

    # Helper: extend a path from both ends greedily
    def extend_path(path):
        if not path:
            return path
        path = path[:]
        extended = True
        while extended:
            extended = False
            # Try extending from the start
            if path:
                first = path[0]
                for nei in adj[first]:
                    if nei not in set(path):
                        path.insert(0, nei)
                        extended = True
                        break
            # Try extending from the end
            if path:
                last = path[-1]
                for nei in adj[last]:
                    if nei not in set(path):
                        path.append(nei)
                        extended = True
                        break
        return path

    # Helper: try to insert a vertex into a path
    def try_insert(path, vertex):
        if vertex in set(path):
            return path
        for i in range(len(path)):
            if i < len(path) - 1:
                if path[i] in adj[vertex] and path[i + 1] in adj[vertex]:
                    new_path = path[:i + 1] + [vertex] + path[i + 1:]
                    return new_path
            elif i == 0:
                if path[0] in adj[vertex]:
                    new_path = [vertex] + path
                    return new_path
            elif i == len(path) - 1:
                if path[-1] in adj[vertex]:
                    new_path = path + [vertex]
                    return new_path
        return path

    # Helper: local search to improve a path
    def local_search(path, time_limit=5.0):
        if not path:
            return path
        path = extend_path(path)
        current_len = len(path)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # Try inserting each unvisited vertex
            for v in range(n):
                if v not in set(path):
                    new_path = try_insert(path, v)
                    if len(new_path) > current_len:
                        path = new_path
                        current_len = len(path)
                        improved = True
                        break
            # Also try swapping adjacent vertices
            for i in range(len(path) - 2):
                a, b, c = path[i], path[i + 1], path[i + 2]
                if c in adj[a]:
                    new_path = path[:]
                    new_path[i + 1], new_path[i + 2] = new_path[i + 2], new_path[i + 1]
                    if is_valid_path(new_path):
                        path = new_path
                        improved = True
                        break
            # Try 2-opt reversal
            for i in range(len(path) - 1):
                for j in range(i + 2, len(path)):
                    if i == 0 or j == len(path) - 1:
                        # Check if reversal creates a valid path
                        if i == 0 and j == len(path) - 1:
                            # Full reversal is always valid if path is valid
                            new_path = path[:]
                            new_path.reverse()
                            if is_valid_path(new_path):
                                path = new_path
                                improved = True
                                break
                        elif i == 0:
                            # Reverse prefix
                            if path[j + 1] in adj[path[0]]:
                                new_path = path[:][::-1] + path[j + 1:]
                                if is_valid_path(new_path):
                                    path = new_path
                                    improved = True
                                    break
                        elif j == len(path) - 1:
                            # Reverse suffix
                            if path[i - 1] in adj[path[-1]]:
                                new_path = path[:i] + path[i:][::-1]
                                if is_valid_path(new_path):
                                    path = new_path
                                    improved = True
                                    break
                if improved:
                    break
            if improved:
                break
        return extend_path(path)

    # Greedy Warnsdorff with forced-move awareness
    def greedy_warnsdorff(start_node, rng):
        visited = {start_node}
        path = [start_node]
        while True:
            last = path[-1]
            neighbors = [x for x in adj[last] if x not in visited]
            if not neighbors:
                break

            # Check for forced moves: neighbors that would become isolated if not visited now
            forced = []
            optional = []
            for x in neighbors:
                # Count unvisited neighbors of x (excluding last, which is visited)
                unvisited_nei_x = adj[x] - visited
                if len(unvisited_nei_x) == 0:
                    # x is a dead end if we don't go there
                    forced.append(x)
                else:
                    optional.append(x)

            if forced:
                # Pick from forced moves
                next_node = forced[rng.randint(0, len(forced) - 1)]
            else:
                # Warnsdorff: pick neighbor with fewest unvisited neighbors
                next_node = min(optional, key=lambda x: (len(adj[x] - visited), x))

            path.append(next_node)
            visited.add(next_node)

        return path

    # Multiple greedy restarts with different strategies
    def multi_greedy(time_budget=10.0):
        best = []
        best_len = 0
        start = time.time()
        rng = random.Random(42)

        # Try different heuristics
        strategies = [
            'warnsdorff',
            'max_degree',
            'random',
            'min_degree',
        ]

        for strategy in strategies:
            for start_node in largest_comp:
                if time.time() - start > time_budget:
                    break

                if strategy == 'warnsdorff':
                    path = greedy_warnsdorff(start_node, rng)
                elif strategy == 'max_degree':
                    visited = {start_node}
                    path = [start_node]
                    while True:
                        last = path[-1]
                        neighbors = [x for x in adj[last] if x not in visited]
                        if not neighbors:
                            break
                        next_node = max(neighbors, key=lambda x: (len(adj[x]), x))
                        path.append(next_node)
                        visited.add(next_node)
                    path = path
                elif strategy == 'random':
                    visited = {start_node}
                    path = [start_node]
                    while True:
                        last = path[-1]
                        neighbors = [x for x in adj[last] if x not in visited]
                        if not neighbors:
                            break
                        next_node = rng.choice(neighbors)
                        path.append(next_node)
                        visited.add(next_node)
                    path = path
                elif strategy == 'min_degree':
                    visited = {start_node}
                    path = [start_node]
                    while True:
                        last = path[-1]
                        neighbors = [x for x in adj[last] if x not in visited]
                        if not neighbors:
                            break
                        next_node = min(neighbors, key=lambda x: (len(adj[x]), x))
                        path.append(next_node)
                        visited.add(next_node)
                    path = path
                else:
                    continue

                # Extend and local search
                path = extend_path(path)
                if len(path) > best_len:
                    best_len = len(path)
                    best = path[:]

                # Local search on promising paths
                if len(path) >= best_len - 2:
                    improved = local_search(path, time_limit=2.0)
                    if len(improved) > best_len:
                        best_len = len(improved)
                        best = improved[:]

            if time.time() - start > time_budget:
                break

        return best, best_len

    # IDDFS with improved pruning
    def iddfs_search(target_len, time_budget=5.0):
        start = time.time()

        def dfs(node, path, visited):
            if time.time() - start > time_budget:
                return None
            if len(path) == target_len:
                return list(path)

            neighbors = [x for x in adj[node] if x not in visited]
            if not neighbors:
                return None

            # Pruning: if we can't reach target_len even with all remaining nodes
            remaining = len(largest_comp) - len(visited)
            if len(path) + remaining < target_len:
                return None

            # Forced move detection
            forced = []
            optional = []
            for x in neighbors:
                unvisited_nei_x = adj[x] - visited
                if len(unvisited_nei_x) == 0:
                    forced.append(x)
                else:
                    optional.append(x)

            # Sort: forced first, then by Warnsdorff
            forced.sort(key=lambda x: (len(adj[x] - visited), x))
            optional.sort(key=lambda x: (len(adj[x] - visited), x))

            ordered_neighbors = forced + optional

            for x in ordered_neighbors:
                path.append(x)
                visited.add(x)
                res = dfs(x, path, visited)
                if res is not None:
                    return res
                path.pop()
                visited.remove(x)

            return None

        # Try start nodes sorted by degree
        start_nodes = sorted(largest_comp, key=lambda x: len(adj[x]))

        for start in start_nodes:
            if time.time() - start > time_budget:
                return None
            path = [start]
            visited = {start}
            res = dfs(start, path, visited)
            if res is not None:
                return res

        return None

    # Main search loop
    # 1. Multi-greedy to get a good initial lower bound
    best_path, best_len = multi_greedy(time_budget=20.0)

    if time.time() > deadline:
        return best_path

    # 2. Try to improve with IDDFS
    current_target = best_len + 1
    max_possible = len(largest_comp)

    while current_target <= max_possible and time.time() < deadline - 10.0:
        # Time budget per target length: 3 seconds
        found_path = iddfs_search(current_target, time_budget=3.0)

        if found_path:
            # Try to extend the found path
            extended = extend_path(found_path)
            if len(extended) > best_len:
                best_len = len(extended)
                best_path = extended[:]
            current_target = best_len + 1
        else:
            # If we couldn't find a path of length current_target, 
            # it's likely we can't find it at all (or it's too hard for our heuristic)
            break

    # 3. Final local search polish on the best path
    if best_path and time.time() < deadline - 5.0:
        polished = local_search(best_path, time_limit=min(10.0, deadline - time.time() - 2.0))
        if len(polished) > best_len:
            best_len = len(polished)
            best_path = polished[:]

    return best_path
# EVOLVE-BLOCK-END

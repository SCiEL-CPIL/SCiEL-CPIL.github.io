import heapq

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
            
    valid_divisors = [d for d in divisors if d <= target]
    
    # Dijkstra's algorithm
    max_sum = target
    dist = [float('inf')] * (max_sum + 1)
    parent = [None] * (max_sum + 1)
    
    dist[0] = 0
    pq = [(0, 0)]
    
    while pq:
        count, s = heapq.heappop(pq)
        
        if s == target:
            # Reconstruct the path
            path = []
            curr = s
            while curr > 0:
                if parent[curr] is None:
                    break
                prev, d = parent[curr]
                path.append(d)
                curr = prev
            
            if len(path) == len(set(path)):
                return path
            else:
                # If duplicates, we need to find another solution.
                # We can't easily do this with simple Dijkstra.
                # We will break and use a fallback.
                break

        for d in valid_divisors:
            new_s = s + d
            if new_s <= max_sum:
                new_count = count + 1
                if new_count < dist[new_s]:
                    dist[new_s] = new_count
                    parent[new_s] = (s, d)
                    heapq.heappush(pq, (new_count, new_s))
    
    # Fallback: Use the original DFS approach for k=1 to 10
    # This ensures we find a valid distinct solution even if Dijkstra failed due to duplicates
    # (which is highly unlikely)
    
    divisors_set = set(divisors)
    divisors_list = sorted(divisors, reverse=True)
    
    # k=1
    if target in divisors_set:
        return [target]
    
    # k=2
    for i in range(len(divisors_list)):
        d1 = divisors_list[i]
        if d1 >= target:
            continue
        d2 = target - d1
        if d2 in divisors_set and d2 != d1:
            return [d1, d2]
            
    # k=3
    for i in range(len(divisors_list)):
        d1 = divisors_list[i]
        if d1 >= target:
            continue
        for j in range(i + 1, len(divisors_list)):
            d2 = divisors_list[j]
            if d1 + d2 >= target:
                continue
            d3 = target - d1 - d2
            if d3 in divisors_set and d3 != d1 and d3 != d2:
                return [d1, d2, d3]
                
    # k=4
    for i in range(len(divisors_list)):
        d1 = divisors_list[i]
        if d1 >= target:
            continue
        for j in range(i + 1, len(divisors_list)):
            d2 = divisors_list[j]
            if d1 + d2 >= target:
                continue
            for k in range(j + 1, len(divisors_list)):
                d3 = divisors_list[k]
                if d1 + d2 + d3 >= target:
                    continue
                d4 = target - d1 - d2 - d3
                if d4 in divisors_set and d4 != d1 and d4 != d2 and d4 != d3:
                    return [d1, d2, d3, d4]
                    
    # k=5 to 10 using DFS
    def dfs_k(idx, remaining, count, current_list):
        if count == 0:
            return current_list if remaining == 0 else None
        if idx >= len(valid_divisors):
            return None
        for j in range(idx, len(valid_divisors)):
            d = valid_divisors[j]
            if d > remaining:
                continue
            result = dfs_k(j + 1, remaining - d, count - 1, current_list + [d])
            if result is not None:
                return result
        return None

    for k in range(5, 11):
        result = dfs_k(0, target, k, [])
        if result is not None:
            return result
            
    return []

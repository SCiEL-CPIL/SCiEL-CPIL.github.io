    while queue:
        if time.time() > deadline:
            break
            
        rem, mask = queue.popleft()
        
        # Check goal
        if rem == 0:
            # Reconstruct
            path = []
            curr = (rem, mask)
            while curr in parent:
                prev_rem, prev_mask, idx = parent[curr]
                path.append(divisors[idx])
                curr = (prev_rem, prev_mask)
            return path

        for i in range(num_divs):
            if mask & (1 << i):
                continue
            d = divisors[i]
            if d > rem:
                continue
            
            new_rem = rem - d
            new_mask = mask | (1 << i)
            
            if (new_rem, new_mask) not in visited:
                visited.add((new_rem, new_mask))
                parent[(new_rem, new_mask)] = (rem, mask, i)
                queue.append((new_rem, new_mask))

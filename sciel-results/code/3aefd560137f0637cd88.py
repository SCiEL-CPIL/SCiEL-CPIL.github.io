    def is_connected_lift(signs):
        # Returns True if lift is connected (i.e., signed graph is unbalanced)
        color = [-1] * n
        color[0] = 1
        stack = [0]
        while stack:
            u = stack.pop()
            # iterate neighbors of u
            # We need adjacency list of base graph
            for v, idx in adj_list[u]:
                s = signs[idx]
                expected = s * color[u]
                if color[v] == -1:
                    color[v] = expected
                    stack.append(v)
                elif color[v] != expected:
                    return True # Unbalanced
        return False # Balanced
    
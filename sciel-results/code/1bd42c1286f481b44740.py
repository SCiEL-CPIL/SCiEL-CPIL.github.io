            if (j - i) % 3 == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

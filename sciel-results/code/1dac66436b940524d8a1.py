    def local_search_hill(init_colors=None):
        if init_colors is None:
            colors = [random.randint(0, 14) for _ in range(n_edges)]
        else:
            colors = init_colors[:]
            
        # ... rest of logic

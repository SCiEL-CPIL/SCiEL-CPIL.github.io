    def construct():
        n = 35
        edges = []
        
        # 8 disjoint K4s
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        
        # Add 3 new vertices 32, 33, 34
        # Connect 32 to 0 (in K4_0)
        # Connect 33 to 4 (in K4_1)
        # Connect 34 to 8 (in K4_2)
        edges.append([32, 0])
        edges.append([33, 4])
        edges.append([34, 8])
        
        return {'n': n, 'edges': edges}
    
<<<<<<< SEARCH
    # Add complementary deletion strategy
    def complementary_deletion(seed=None):
        """
        Start with all edges, remove edges to break all C4s.
        """
        rng = random.Random(seed)
        sol = [True] * n_edges
        face_counts = [4] * n_faces
        bad_faces = set(range(n_faces))
        
        while bad_faces and time_left() > 5:
            fi_list = list(bad_faces)
            if not fi_list: break
            fi = fi_list[rng.randint(0, len(fi_list)-1)]
            
            face_edges = faces[fi]
            ei = rng.choice(face_edges)
            
            if sol[ei]:
                sol[ei] = False
                for f2 in edge_faces[ei]:
                    face_counts[f2] -= 1
                    if face_counts[f2] == 3:
                        bad_faces.discard(f2)
        
        return sol
=======
    # Add complementary deletion strategy with priority-based edge selection
    def complementary_deletion(seed=None):
        """
        Start with all edges, remove edges to break all C4s.
        Uses priority: remove edges that appear in the most bad faces.
        """
        rng = random.Random(seed)
        sol = [True] * n_edges
        face_counts = [4] * n_faces
        bad_faces = set(range(n_faces))
        
        while bad_faces and time_left() > 5:
            # For each edge, count how many bad faces it's in
            edge_bad_count = {}
            for fi in bad_faces:
                for ei in faces[fi]:
                    if sol[ei]:
                        edge_bad_count[ei] = edge_bad_count.get(ei, 0) + 1
            
            if not edge_bad_count:
                break
            
            # Sort edges by bad face count (descending), with random tie-breaking
            sorted_edges = sorted(edge_bad_count.items(), key=lambda x: (-x[1], rng.random()))
            
            # Pick the top edge (or randomly from top few for diversity)
            top_k = min(3, len(sorted_edges))
            ei = sorted_edges[rng.randint(0, top_k - 1)][0]
            
            sol[ei] = False
            for f2 in edge_faces[ei]:
                face_counts[f2] -= 1
                if face_counts[f2] == 3:
                    bad_faces.discard(f2)
        
        return sol
>>>>>>> REPLACE


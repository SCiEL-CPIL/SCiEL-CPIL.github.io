    def solve_scope(scope, seed, max_nodes=...):
        # State:
        # used_bits: global
        # row_data: list of dicts or tuples for each row
        #   each row has: current_list, min_x, local_diff_bits
        #   status: 'incomplete', 'complete', 'failed'
        
        def get_candidates(row_idx):
            # Compute valid x for row_idx
            pass
        
        def search():
            # Find row with min candidates
            # If any row has 0 candidates, fail
            # Pick a candidate
            # Recurse
            pass
    
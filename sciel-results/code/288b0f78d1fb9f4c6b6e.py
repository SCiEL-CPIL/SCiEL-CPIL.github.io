<<<<<<< SEARCH
def _backtrack_scope(rows, k, scope, deadline):
    """
    Deterministic refinement: try to rebuild the construction with every entry
    <= scope. Returns a valid list of rows if found, else None.
    Uses a shared used-diff set across rows and backtracks per row.
    """
    used = set()
    rows_out = []

    def build_row(used):
=======
def _backtrack_scope(rows, k, scope, deadline):
    """
    Deterministic refinement: try to rebuild the construction with every entry
    <= scope. Returns a valid list of rows if found, else None.
    Uses a shared used-diff set across rows and backtracks per row.
    """
    n_rows = len(rows)
    used = set()
    rows_out = []

    def build_row(used):
>>>>>>> REPLACE

Also, the deadline in `construct` should be tightened to ensure we don't hit the hard limit. The contract says 120s. The prompt says 150s kill. Let's use 110s to be safe.


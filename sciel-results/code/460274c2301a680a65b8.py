<<<<<<< SEARCH
"""
Combined difference-triangle solver.

Strategy:
  Phase 1 – Fast randomised greedy with look-ahead (many restarts).
  Phase 2 – Simulated-annealing local search on individual entries.
  Phase 3 – DFS row re-optimisation: for each row (in size order), try to
            minimise its max entry given the other rows are fixed.  This
            escapes greedy basins and pushes scope toward the best-known value.
  Phase 4 – Final polish pass of the same re-optimisation.

All phases are bounded by an explicit wall-clock deadline (140 s) so the
program always returns the best result found.
"""

import time
import random
import sys
=======
"""
Difference-triangle solver using row-level replacement search.

Strategy:
  Phase 1 – Randomised greedy with variable look-ahead (many restarts).
  Phase 2 – Row-level simulated annealing: replace entire rows with
            newly-built rows, using incremental diff-count validation
            for O(1) amortized feasibility checks.
  Phase 3 – Targeted row re-optimisation via DFS.
  Phase 4 – Final polish.

All phases are bounded by an explicit wall-clock deadline (138 s) so the
program always returns the best result found.
"""

import time
import random
import sys
>>>>>>> REPLACE


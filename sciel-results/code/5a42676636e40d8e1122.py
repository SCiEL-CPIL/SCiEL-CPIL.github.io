<<<<<<< SEARCH
    def validate_routes(routes):
        """Check if routes are valid: all customers served exactly once, all feasible."""
        all_customers = []
        for route in routes:
            feasible, _ = check_route(route)
            if not feasible:
                return False
            all_customers.extend(route)
        if len(all_customers) != len(set(all_customers)):
            return False
        if set(all_customers) != set(customers):
            return False
        return True
=======
    def validate_routes(routes):
        """Check if routes are valid: all customers served exactly once, all feasible."""
        all_customers = []
        for route in routes:
            feasible, _, _ = check_route(route)
            if not feasible:
                return False
            all_customers.extend(route)
        if len(all_customers) != len(set(all_customers)):
            return False
        if set(all_customers) != set(customers):
            return False
        return True
>>>>>>> REPLACE

Update `total_route_distance` to use `check_route`? No, `total_route_distance` is used in local search for incremental updates.
But now we are using `get_true_distance` which calls `check_route` for every route. This is slow.
In local search, we should use `fast_check_route` for filtering and then `check_route` for the accepted move.
But we need the current distance to compare.
We can maintain `best_distance` and only call `check_route` on the new candidate.

Let's update `two_opt_route` to work with customer lists and use `check_route` only when a move is promising.

Actually, the `two_opt_route` I wrote above extracts customers, tries a move, does `fast_check`, then `check_route`.
It updates `route` to the full route with stations.
But the next iteration of 2-opt needs to work on the customer list again.
So I should return the customer list, not the full route?
Or keep the full route and extract customers every time?
Extracting customers is O(n). That's fine.

But the local search functions `relocate_search` and `cross-route swap` also need updating.
They currently operate on `routes` which are lists of node IDs (customers).
If I keep them as customer lists, I need to update the distance calculation.

Let's simplify:
1. `plan` maintains `best_routes` as list of customer lists.
2. `best_distance` is the true distance (with charging).
3. Local search functions take customer lists, try moves, use `fast_check` + `check_route` to validate and get new distance.
4. If improved, update the customer list and `best_distance`.

This requires rewriting the local search functions to be customer-list based.

Given the complexity, let's just replace the whole EVOLVE-BLOCK with a cleaner implementation that follows this structure.


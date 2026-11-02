# SCiEL

The project page of SCiEL, recursive self-improvement through cultural transmission:
https://sciel-rsi-cpil.github.io/SCiEL-CPIL.github.io/

A static page served by GitHub Pages from `main`. The code is at
[Doldrums/sciel](https://github.com/Doldrums/sciel); the runs shown on the page are in the
[Empira arena](https://research-arena.small-pine-3af7.workers.dev/experiments).

## Layout

- `index.html` – the page.
- `assets/style.css` – the styles.
- `assets/runs.js` – the six runs the figures are drawn from, exported from the run viewer.
- `assets/*.js` – one file per figure or section, loaded in the order listed in `index.html`:
  `page` (helpers, navigation, run selector, tooltips), `disk`, `cost`, `replay`, `culture`,
  `progress`, `records`, `world` (the toy landscape and the four searchers), `compare-scene`,
  `compare`, `niches`, `relief`.

## Working on it

```sh
python3 -m http.server 8000
```

Then open http://localhost:8000/. Keep each file at or below 500 lines, and keep comments out of
the code; the README is the only Markdown file in the repository. Use Conventional Commit messages.
